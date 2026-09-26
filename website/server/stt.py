"""Speech-to-text via faster-whisper (CTranslate2).

The model is loaded once and reused. Device selection is deliberately
defensive: this box is a Blackwell (sm_120) card, so CUDA support is probed at
load time and we fall back to CPU int8 rather than failing to start.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .cuda_compat import ensure_cuda_runtime

log = logging.getLogger("friday.stt")

# Whisper reliably emits spelled-out numbers and operators. Normalising them
# here means the calc tool and the LLM both see "7 times 23" instead of
# "seven times twenty three".
_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
    "ninety": "90",
}
_OP_WORDS = {
    # Multi-word forms first: "divided by" must not leave a stray "by" behind.
    "divided by": "/",
    "multiplied by": "*",
    "raised to the power of": "**",
    "to the power of": "**",
    "percent of": "%",
    # Single-word fallbacks.
    "plus": "+", "minus": "-", "times": "*", "multiplied": "*", "divided": "/",
    "over": "/", "power": "**", "squared": "** 2",
    "cubed": "** 3", "modulo": "%", "remainder": "%",
}
# Tokens stripped from the *front* of a transcript. Deliberately conservative:
# "hello" and "hey" are legitimate content ("hello there", "hey, what is the
# weather"), so only the wake word and hesitation fillers are removed. The
# firmware should already have cut the wake word before recording.
LEADING_FILLERS: tuple[str, ...] = ("friday", "um", "uh", "erm")
_FILLER = re.compile(
    r"^\s*(?:" + "|".join(LEADING_FILLERS) + r")\b[\s,]*", re.IGNORECASE
)
# "hey friday, what is ..." -- a greeting in front of a real filler. Stripped only
# when something strippable follows, so a bare "hey, what is the weather" keeps
# its greeting and "hey friday" does not linger.
_GREETING_BEFORE_FILLER = re.compile(
    r"^\s*(?:hey|hi|hello|ok|okay|good\s+morning|good\s+evening)\b[\s,]*"
    r"(?=(?:" + "|".join(LEADING_FILLERS) + r")\b)",
    re.IGNORECASE,
)

# Number scales, applied left to right: "two hundred" -> 200.
_SCALES: dict[str, int] = {
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
}
# Whisper writes "twenty-three"; a hyphen between letters joins a compound
# number. A hyphen between digits is subtraction, so only letters qualify.
_HYPHEN_JOINER = re.compile(r"(?<=[a-zA-Z])-(?=[a-zA-Z])")
# Words and bare numbers, for repetition detection. Punctuation is dropped so
# "Kolkata, Kolkata. Kolkata!" reads as three in a row rather than three words.
_WORD_RE = re.compile(r"[a-z0-9']+")


@dataclass(slots=True)
class Transcript:
    text: str
    confidence: float
    duration_s: float
    elapsed_ms: int
    language: str = "en"
    # Words Whisper was unsure about, useful when a query comes back odd.
    low_confidence: list[str] | None = None
    # True when the transcript is a decode loop rather than speech. Whisper
    # produces these at *high* confidence, so `confidence` cannot catch them.
    repetitive: bool = False


class SpeechToText:
    """Thin wrapper around ``faster_whisper.WhisperModel``."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._model = None
        self.device: str = "unloaded"
        self.compute_type: str = "unloaded"
        self._load_lock: object = None

    # ------------------------------------------------------------------ load
    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the model, degrading to CPU if the GPU path is unavailable."""
        if self._model is not None:
            return

        # Must happen before CTranslate2 dlopens cuBLAS/cuDNN.
        ensure_cuda_runtime()

        from faster_whisper import WhisperModel

        device, compute_type = self._resolve_target()
        log.info("loading STT model %s on %s (%s)", self.cfg.stt.model, device, compute_type)
        try:
            self._model = WhisperModel(
                self._resolve_model_ref(),
                device=device,
                compute_type=compute_type,
            )
            self.device, self.compute_type = device, compute_type
        except Exception as exc:  # noqa: BLE001
            if device == "cuda":
                log.warning("CUDA init failed (%s); falling back to CPU int8", exc)
                self._model = WhisperModel(
                    self._resolve_model_ref(), device="cpu", compute_type="int8"
                )
                self.device, self.compute_type = "cpu", "int8"
            else:
                raise

    def _resolve_model_ref(self) -> str:
        """Allow a local directory (absolute, project-relative, or in ``models/``)
        or a Hugging Face model id.

        Checking ``models/`` matters: `scripts/fetch_models.py` prefetches into
        that directory, and a bare hub id like ``small.en`` would otherwise make
        faster-whisper re-download (or hang offline) even though the weights are
        already on disk.
        """
        ref = self.cfg.stt.model
        direct = Path(ref)
        if direct.exists():
            return str(direct.resolve())
        local = self.cfg.models_dir / ref
        if local.is_dir():
            return str(local.resolve())
        # A hub id that names a local cache directory, e.g. "faster-whisper-small.en".
        alias = self.cfg.models_dir / f"faster-whisper-{ref}"
        if alias.is_dir():
            return str(alias.resolve())
        return ref

    def _resolve_target(self) -> tuple[str, str]:
        want = self.cfg.stt.device
        compute = self.cfg.stt.compute_type

        if want == "cpu":
            return "cpu", "int8" if compute == "auto" else compute
        if want == "cuda":
            return "cuda", self._cuda_compute_type(compute)

        # auto
        if _cuda_available():
            return "cuda", self._cuda_compute_type(compute)
        log.info("no CUDA device detected; STT will run on CPU")
        return "cpu", "int8"

    @staticmethod
    def _cuda_compute_type(compute: str) -> str:
        if compute != "auto":
            return compute
        # int8 quantisation of the encoder keeps VRAM small and is faster on
        # consumer Blackwell cards than float16 for this model size.
        try:
            import ctranslate2

            supported = ctranslate2.get_supported_compute_types("cuda")
            for candidate in ("int8_float16", "int8", "float16", "float32"):
                if candidate in supported:
                    return candidate
        except Exception:  # noqa: BLE001
            pass
        return "int8_float16"

    # ----------------------------------------------------------- transcribe
    def transcribe(self, audio_path: Path) -> Transcript:
        """Transcribe a WAV file."""
        self.load()
        assert self._model is not None

        started = time.monotonic()
        segments, info = self._model.transcribe(
            str(audio_path),
            beam_size=self.cfg.stt.beam_size,
            vad_filter=self.cfg.stt.vad_filter,
            # Keep hallucinated silence out of the transcript.
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
            temperature=0.0,
            hotwords=" ".join(self.cfg.stt.hotwords) or None,
            initial_prompt=self.cfg.stt.initial_prompt,
        )

        parts: list[str] = []
        weighted = 0.0
        weight = 0.0
        low: list[str] = []
        audio_dur = 0.0
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            parts.append(text)
            avg = getattr(seg, "avg_logprob", 0.0)
            conf = _logprob_to_conf(avg)
            weighted += conf * len(text)
            weight += len(text)
            if conf < self.cfg.stt.min_confidence:
                low.append(text)
            audio_dur += seg.end - seg.start

        raw = " ".join(parts).strip()
        confidence = (weighted / weight) if weight else 0.0
        elapsed_ms = int((time.monotonic() - started) * 1000)

        text = normalize_transcript(raw) if self.cfg.stt.normalize_numbers else raw
        log.info(
            "transcribed %s -> %r (conf %.2f, %dms)", audio_path.name, text, confidence, elapsed_ms
        )
        return Transcript(
            text=text,
            confidence=round(confidence, 3),
            duration_s=round(audio_dur, 2),
            elapsed_ms=elapsed_ms,
            language=getattr(info, "language", "en") or "en",
            low_confidence=low or None,
            repetitive=is_repetitive(text),
        )


# How many times the same word may repeat back to back before the transcript is
# treated as a decode loop. Calibrated against real captures: the nine genuine
# commands never exceed a run of two ("What is 2 + 2"), while Whisper looping
# over music ran "Kolkata" eighteen deep at confidence 0.73.
_REPEAT_RUN = 3


def is_repetitive(text: str) -> bool:
    """True when a word repeats back to back, which speech does not do.

    Whisper's signature failure on music, noise and room tone is a repetition
    loop, and it is the reason confidence cannot be the only gate: the model is
    genuinely *sure* of the phrase it is stuck on. A real ESP32 capture of a
    speaker playing music scored 0.73 with twelve consecutive copies of one
    place name, which sails through any confidence threshold and would otherwise
    be answered as if it were a question.
    """
    toks = _WORD_RE.findall(text.lower())
    run = 0
    previous = ""
    for tok in toks:
        run = run + 1 if tok == previous else 1
        if run >= _REPEAT_RUN:
            return True
        previous = tok
    return False


def _logprob_to_conf(avg_logprob: float) -> float:
    """Map Whisper's avg_logprob onto a 0..1 confidence."""
    import math

    if avg_logprob is None:
        return 0.0
    try:
        return max(0.0, min(1.0, math.exp(avg_logprob)))
    except (OverflowError, ValueError):
        return 0.0


def normalize_transcript(text: str) -> str:
    """Tidy a raw transcript: strip fillers, spell out digits and operators."""
    if not text:
        return ""
    s = _HYPHEN_JOINER.sub(" ", text)
    # Whisper often stacks fillers: "Hey Friday, what is ...". Peel them all.
    for _ in range(6):
        stripped = _FILLER.sub("", s).strip()
        stripped = _GREETING_BEFORE_FILLER.sub("", stripped).strip()
        if stripped == s:
            break
        s = stripped
    s = re.sub(r"\s+", " ", s)

    lowered = s.lower()
    for phrase, sym in sorted(_OP_WORDS.items(), key=lambda kv: -len(kv[0])):
        pattern = r"\b" + re.escape(phrase) + r"\b"
        if re.search(pattern, lowered):
            s = re.sub(pattern, f" {sym} ", s, flags=re.IGNORECASE)
            lowered = s.lower()

    # Compound numbers first ("twenty three" -> "23"), then scales, then the
    # leftover single words. The order matters: "two hundred and fifty" has to
    # become 200 before "fifty" can be replaced.
    tens = "|".join(w for w, _ in _NUM_WORDS.items() if int(_NUM_WORDS[w]) >= 20)
    units = "|".join(w for w, _ in _NUM_WORDS.items() if 0 < int(_NUM_WORDS[w]) < 10)
    if tens:
        s = re.sub(
            rf"\b({tens})\s+({units})\b",
            lambda m: str(int(_NUM_WORDS[m.group(1)]) + int(_NUM_WORDS[m.group(2)])),
            s,
            flags=re.IGNORECASE,
        )
    s = re.sub(
        r"\b([a-z]+)\b",
        lambda m: _NUM_WORDS.get(m.group(1).lower(), m.group(1)),
        s,
        flags=re.IGNORECASE,
    )
    for name, multiplier in _SCALES.items():
        s = re.sub(
            rf"\b(\d+)\s+{name}\b",
            lambda m, k=multiplier: str(int(m.group(1)) * k),
            s,
            flags=re.IGNORECASE,
        )
    # "200 and 50" is now redundant; leaving the word in would reach the LLM.
    s = re.sub(r"\b(\d+)\s+and\s+(\d+)\b", r"\1 \2", s)
    return re.sub(r"\s+", " ", s).strip()


def _cuda_available() -> bool:
    """Cheap, non-throwing CUDA probe."""
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001
        return False
