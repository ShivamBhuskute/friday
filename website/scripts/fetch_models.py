#!/usr/bin/env python
"""Prefetch model weights into ./models so the first run is offline and fast."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STT_REPO = "Systran/faster-whisper-small.en"
LLM_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
LLM_FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"


def fetch_stt(target: Path) -> Path:
    """Download the CTranslate2 Whisper model into ``target``."""
    if target.exists() and (target / "model.bin").exists():
        print(f"[skip] STT model already present at {target}")
        return target
    print(f"[....] downloading {STT_REPO} -> {target}")
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=STT_REPO,
        local_dir=str(target),
        allow_patterns=["*.json", "*.bin", "*.txt"],
    )
    print(f"[ok  ] STT model at {target}")
    return target


def fetch_llm(target: Path) -> Path:
    """Download the Qwen GGUF into ``target``."""
    if target.exists() and target.stat().st_size > 100_000_000:
        print(f"[skip] LLM already present at {target}")
        return target
    print(f"[....] downloading {LLM_REPO}/{LLM_FILE} -> {target}")
    from huggingface_hub import hf_hub_download

    target.parent.mkdir(parents=True, exist_ok=True)
    got = hf_hub_download(repo_id=LLM_REPO, filename=LLM_FILE)
    # hf_hub_download returns a cache path; link it into ./models.
    link = target.resolve()
    if Path(got).resolve() != link:
        if link.exists() or link.is_symlink():
            link.unlink()
        try:
            link.symlink_to(Path(got).resolve())
        except OSError:
            import shutil

            shutil.copy2(got, link)
    print(f"[ok  ] LLM at {link}")
    return link


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stt", action="store_true", help="only the STT model")
    ap.add_argument("--llm", action="store_true", help="only the LLM")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    do_stt = args.stt or not args.llm
    do_llm = args.llm or not args.stt

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    if args.force:
        for p in models_dir.glob("*"):
            if p.is_symlink():
                p.unlink()
            elif p.is_dir():
                import shutil

                shutil.rmtree(p)
            else:
                p.unlink()

    # Keep HF cache inside the project so the 14GB disk stays predictable.
    os.environ.setdefault("HF_HOME", str(models_dir / ".hf-cache"))

    try:
        if do_stt:
            fetch_stt(models_dir / "faster-whisper-small.en")
        if do_llm:
            fetch_llm(models_dir / LLM_FILE)
    except ImportError:
        print(
            "huggingface_hub is required. Install it with:\n"
            "  uv pip install huggingface_hub",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"model download failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("\nAll models ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
