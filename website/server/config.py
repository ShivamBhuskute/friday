"""Configuration loading for the FRIDAY pipeline.

Values come from ``config.yaml`` at the project root, optionally overridden by a
git-ignored ``config.local.yaml``, with environment variable overrides using
the ``FRIDAY_`` prefix and ``__`` for nesting. Precedence, lowest to highest:
``config.yaml``, ``config.local.yaml``, environment.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

log = logging.getLogger("friday.config")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    serve_web: bool = True


class IngestConfig(BaseModel):
    port: int = 5000
    host: str = "0.0.0.0"
    sample_rate: int = 16000
    channels: int = 1
    bits: int = 16
    vad_silence_s: float = 0.8
    # Must stay below the device's STREAM_SILENCE_RMS_GATE (0.007) -- the
    # device does its own endpointing, so this is only a safety net. See
    # server/ingest/protocol.py.
    vad_rms_threshold: float = 0.006
    max_utterance_s: float = 10.0
    min_utterance_s: float = 0.3
    max_connections: int = 4


class SttConfig(BaseModel):
    model: str = "small.en"
    device: Literal["auto", "cuda", "cpu"] = "auto"
    compute_type: Literal["auto", "int8_float16", "int8", "float16", "float32"] = "auto"
    beam_size: int = 5
    vad_filter: bool = True
    normalize_numbers: bool = True
    # Calibrated against the firmware's own recordings in
    # friday_kws/deteced_recordings/: the nine genuine commands score 0.62-0.83,
    # while false wake-word activations on music and room tone scored 0.28-0.45.
    # 0.5 keeps every real command with margin and drops all of those.
    min_confidence: float = 0.5
    # Vocabulary biasing. small.en mishears proper nouns ("Pune" -> "Pooner");
    # biasing the decoder with the words this assistant actually uses fixes it
    # without a larger model.
    hotwords: list[str] = Field(default_factory=list)
    initial_prompt: str | None = None


class LlmConfig(BaseModel):
    model_path: str = "models/qwen2.5-3b-instruct-q4_k_m.gguf"
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    max_tokens: int = 400
    temperature: float = 0.2
    max_tool_rounds: int = 4
    eager_load: bool = True
    # llama-cpp-python chat handler. Qwen2.5 needs the function-calling variant;
    # plain "chatml" ignores the tools and hallucinates the arithmetic instead.
    chat_format: str = "chatml-function-calling"


class ToolsConfig(BaseModel):
    weather_cache_ttl: float = 600.0
    weather_timeout_s: float = 8.0
    calc_fast_path: bool = True


class RetentionConfig(BaseModel):
    max_turns: int = 200
    delete_audio: bool = True


class Config(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)

    models_dir: Path = PROJECT_ROOT / "models"
    data_dir: Path = PROJECT_ROOT / "data"

    # ---------------------------------------------------------------- paths
    @property
    def recordings_dir(self) -> Path:
        return self.data_dir / "recordings"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "friday.db"

    @property
    def sysmon_path(self) -> Path:
        """Optional device-written system monitor snapshot."""
        return self.data_dir / "sysmon.json"

    @property
    def llm_model_file(self) -> Path:
        p = Path(self.llm.model_path)
        return p if p.is_absolute() else self.models_dir / p.name

    def ensure_dirs(self) -> None:
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------- loading
    @classmethod
    def load(cls, path: Path | str | None = None) -> Config:
        """Load config, then apply ``FRIDAY_*`` env overrides.

        Precedence, lowest to highest: ``config.yaml``, ``config.local.yaml``,
        environment. The local file is merged one level deep, so it can change a
        single knob (``ingest: {port: 9000}``) without restating the rest, and
        it is git-ignored so a per-machine port or device IP can be committed
        nowhere.
        """
        cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
        raw: dict[str, Any] = _load_yaml(cfg_path)
        raw = _deep_merge(raw, _load_yaml(PROJECT_ROOT / "config.local.yaml"))

        # A couple of top-level keys are convenience aliases, not sections.
        models_dir = raw.pop("models", {}) or {}
        data_dir = raw.pop("data", {}) or {}

        cfg = cls.model_validate(raw)

        if isinstance(models_dir, dict) and "dir" in models_dir:
            cfg.models_dir = _resolve(models_dir["dir"])
        cfg.data_dir = _resolve(data_dir.get("dir", "data") if isinstance(data_dir, dict) else data_dir)

        _apply_env_overrides(cfg)
        return cfg


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError:
        log.warning("ignoring %s: not valid YAML", path)
        return {}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge ``overlay`` onto ``base``, one level into each section.

    A deep merge is tempting but wrong here: a partially-specified nested block
    (a list of hotwords, a sub-model) would silently inherit siblings from the
    other file. Merging only the section level means "replace this section
    whole", which is what a person editing a local override expects.
    """
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = {**existing, **value}
        else:
            merged[key] = value
    return merged


def _resolve(value: str | os.PathLike[str]) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _coerce(current: Any, raw: str) -> Any:
    """Coerce an env string to the type of the current config value."""
    if isinstance(current, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def _apply_env_overrides(cfg: Config) -> None:
    """Apply ``FRIDAY_SECTION__KEY=value`` overrides, e.g. ``FRIDAY_STT__DEVICE=cpu``."""
    for env_key, env_val in os.environ.items():
        if not env_key.startswith("FRIDAY_") or "__" not in env_key:
            continue
        _, _, path = env_key.partition("FRIDAY_")
        parts = path.lower().split("__")
        target: Any = cfg
        for part in parts[:-1]:
            target = getattr(target, part, None)
            if target is None:
                break
        if target is None:
            continue
        field = parts[-1]
        if hasattr(target, field):
            setattr(target, field, _coerce(getattr(target, field), env_val))


_cached: Config | None = None


def get_config() -> Config:
    """Process-wide singleton config."""
    global _cached
    if _cached is None:
        _cached = Config.load()
    return _cached
