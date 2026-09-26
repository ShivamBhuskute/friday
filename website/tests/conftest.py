"""Shared pytest fixtures."""

from __future__ import annotations

import asyncio
import math
import struct
from pathlib import Path

import pytest

from server.bus import EventBus
from server.config import Config
from server.db import Database
from server.ingest.wav import build_wav_header

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = PROJECT_ROOT / "fixtures"
MODELS = PROJECT_ROOT / "models"

STT_MODEL_DIR = MODELS / "faster-whisper-small.en"
LLM_FILE = MODELS / "qwen2.5-3b-instruct-q4_k_m.gguf"


def has_stt_model() -> bool:
    return (STT_MODEL_DIR / "model.bin").exists()


def has_llm_model() -> bool:
    return LLM_FILE.exists()


needs_stt = pytest.mark.skipif(not has_stt_model(), reason="STT weights not downloaded")
needs_llm = pytest.mark.skipif(not has_llm_model(), reason="LLM weights not downloaded")


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """A config pointed at a throwaway data dir, with models left in place."""
    config = Config.load()
    config.data_dir = tmp_path / "data"
    config.models_dir = MODELS
    config.stt.model = str(STT_MODEL_DIR) if STT_MODEL_DIR.exists() else "small.en"
    config.llm.model_path = str(LLM_FILE) if LLM_FILE.exists() else "qwen2.5-3b-instruct-q4_k_m.gguf"
    config.llm.eager_load = False
    config.ensure_dirs()
    return config


@pytest.fixture(scope="session")
def session_cfg(tmp_path_factory) -> Config:
    """Same as :func:`cfg` but shared, so a model is only loaded once per run."""
    config = Config.load()
    config.data_dir = tmp_path_factory.mktemp("session-data")
    config.models_dir = MODELS
    config.stt.model = str(STT_MODEL_DIR) if STT_MODEL_DIR.exists() else "small.en"
    config.llm.model_path = str(LLM_FILE) if LLM_FILE.exists() else "qwen2.5-3b-instruct-q4_k_m.gguf"
    config.llm.eager_load = False
    config.ensure_dirs()
    return config


@pytest.fixture
def db(cfg: Config) -> Database:
    database = Database(cfg.db_path)
    yield database
    database.close()


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


# ------------------------------------------------------------------ audio


def tone_pcm(
    seconds: float,
    *,
    sample_rate: int = 16000,
    freq: float = 220.0,
    amplitude: int = 9000,
) -> bytes:
    """A mono 16-bit sine, standing in for a voice when shape is all that matters."""
    n = int(seconds * sample_rate)
    samples = (
        int(amplitude * math.sin(2 * math.pi * freq * i / sample_rate)) for i in range(n)
    )
    return struct.pack(f"<{n}h", *samples)


def silence_pcm(seconds: float, *, sample_rate: int = 16000) -> bytes:
    return b"\x00\x00" * int(seconds * sample_rate)


def wav_bytes(
    pcm: bytes,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    bits: int = 16,
) -> bytes:
    return (
        build_wav_header(
            sample_rate=sample_rate,
            channels=channels,
            bits_per_sample=bits,
            data_size=len(pcm),
        )
        + pcm
    )


@pytest.fixture
def make_wav():
    return wav_bytes


@pytest.fixture
def tone():
    return tone_pcm


@pytest.fixture(scope="session")
def event_loop_policy():
    return asyncio.DefaultEventLoopPolicy()
