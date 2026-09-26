#!/usr/bin/env python
"""Phase 0 gate: prove CUDA inference actually works on this machine.

Both inference engines have to work on the GPU, and both have a CPU fallback,
so this script reports what is available and exits non-zero only when nothing
usable is found. Run it before trusting any latency number.
"""

from __future__ import annotations

import argparse
import sys


def check_ct2() -> tuple[bool, str]:
    try:
        import ctranslate2
    except ImportError:
        return False, "ctranslate2 not installed"
    try:
        n = ctranslate2.get_cuda_device_count()
    except Exception as exc:  # noqa: BLE001
        return False, f"CUDA probe failed: {exc}"
    if n < 1:
        return False, "no CUDA devices visible to CTranslate2"
    try:
        kinds = sorted(ctranslate2.get_supported_compute_types("cuda"))
    except Exception:  # noqa: BLE001
        kinds = []
    return True, f"{n} device(s); cuda compute types: {', '.join(kinds) or 'unknown'}"


def check_llama_cpp() -> tuple[bool, str]:
    try:
        import llama_cpp
    except ImportError:
        return False, "llama-cpp-python not installed"
    try:
        lib = llama_cpp.llama_cpp  # noqa: B018 - probing that the shared lib loaded
    except Exception as exc:  # noqa: BLE001
        return False, f"native library failed to load: {exc}"
    del lib
    try:
        supports = llama_cpp.llama_supports_gpu_offload()
    except Exception as exc:  # noqa: BLE001
        return False, f"gpu offload probe failed: {exc}"
    return True, f"loaded; gpu_offload={bool(supports)}"


def check_torch_free_cuda() -> tuple[bool, str]:
    """Report the GPU identity without requiring torch to be installed."""
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return True, out.stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return True, "nvidia-smi unavailable (not fatal)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--require-gpu",
        action="store_true",
        help="exit non-zero unless both engines report GPU support",
    )
    args = ap.parse_args()

    print("FRIDAY GPU verification")
    print("=" * 60)

    results: dict[str, tuple[bool, str]] = {}
    for name, fn in (
        ("gpu", check_torch_free_cuda),
        ("ctranslate2", check_ct2),
        ("llama-cpp", check_llama_cpp),
    ):
        ok, detail = fn()
        results[name] = (ok, detail)
        print(f"[{'ok ' if ok else 'FAIL'}] {name:14} {detail}")

    print("=" * 60)
    gpu_engines = [results["ctranslate2"][0], results["llama-cpp"][0]]

    if all(gpu_engines):
        print("Both engines report GPU support.")
    elif any(gpu_engines):
        print("Partial GPU support. Set the failing engine to CPU in config.yaml:")
        if not results["ctranslate2"][0]:
            print("  stt.device: cpu")
        if not results["llama-cpp"][0]:
            print("  llm.n_gpu_layers: 0")
    else:
        print("No GPU acceleration. Running on CPU (slower but functional).")

    if args.require_gpu and not all(gpu_engines):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
