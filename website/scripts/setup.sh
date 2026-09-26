#!/usr/bin/env bash
#
# One-shot setup: venv, CUDA-enabled llama-cpp-python, model weights, frontend.
#
# Idempotent -- safe to re-run. The llama-cpp-python build is the slow part
# (~5 minutes from sdist); everything else is a few minutes of downloads.
#
#   ./scripts/setup.sh               # everything
#   ./scripts/setup.sh --no-llama    # skip the CUDA build (CPU-only dev box)
#   ./scripts/setup.sh --no-web      # skip the frontend
#   ./scripts/setup.sh --no-models   # skip the weight download
#
# NOTE: keep the usage block above in sync with the option parsing below.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
PY="$VENV/bin/python"
BUILD_LLAMA=1
BUILD_WEB=1
FETCH_MODELS=1

for arg in "$@"; do
  case "$arg" in
    --no-llama)  BUILD_LLAMA=0 ;;
    --no-web)    BUILD_WEB=0 ;;
    --no-models) FETCH_MODELS=0 ;;
    -h|--help)   sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- interpreter
say "Checking the toolchain"
if command -v uv >/dev/null 2>&1; then
  echo "uv      $(uv --version)"
elif [ -x "$HOME/.local/bin/uv" ]; then
  export PATH="$HOME/.local/bin:$PATH"
  echo "uv      $(uv --version)"
else
  die "uv not found. Install it:  curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "gpu     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
  warn "no NVIDIA GPU visible -- STT will fall back to CPU int8, and be slow"
fi

# ---------------------------------------------------------------------- venv
say "Creating the virtualenv (.venv, Python 3.13)"
if [ ! -x "$PY" ]; then
  uv venv --python 3.13 "$VENV"
else
  echo "already exists: $VENV"
fi

# Keep every byte of model and cache data inside the project, so the disk
# footprint is one number instead of three.
export HF_HOME="$ROOT/models/.hf-cache"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.cache/uv}"
mkdir -p "$ROOT/models" "$HF_HOME"

# ---------------------------------------------------------- base dependencies
say "Installing Python dependencies"
uv pip install --python "$PY" -e ".[dev]"

# ------------------------------------------------- llama-cpp-python (CUDA build)
if [ "$BUILD_LLAMA" -eq 1 ]; then
  say "Building llama-cpp-python with CUDA (this takes several minutes)"
  # There is no CUDA wheel on PyPI, so this compiles from sdist. Without
  # -DGGML_CUDA=on the LLM runs on CPU and a 3B model answers in ~30s.
  CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_BUILD_PARALLEL_LEVEL=$(nproc)" \
    uv pip install --python "$PY" --no-cache-dir --force-reinstall \
    --no-binary llama-cpp-python "llama-cpp-python>=0.3.35"
else
  say "Skipping the llama-cpp-python CUDA build (--no-llama)"
  uv pip install --python "$PY" --no-binary llama-cpp-python "llama-cpp-python>=0.3.35"
  warn "the LLM will run on CPU: expect multi-second answers"
fi

# -------------------------------------------------------------- CUDA runtime
say "Checking the CUDA runtime that CTranslate2 needs"
# CTranslate2 links libcublas.so.12 / libcudnn.so.9, which a CUDA 13 toolkit does
# not provide; server/cuda_compat.py preloads the cu12 wheels to bridge that.
"$PY" - <<'PYCHECK' || warn "could not confirm the CUDA runtime; STT may fall back to CPU"
import ctypes, sys
for name in ("libcublas.so.12", "libcudnn.so.9"):
    try:
        ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        print(f"  missing {name}: {exc}")
        sys.exit(1)
    print(f"  ok {name}")
PYCHECK

# --------------------------------------------------------------------- models
if [ "$FETCH_MODELS" -eq 1 ]; then
  say "Fetching model weights (about 2.5 GB)"
  "$PY" scripts/fetch_models.py
else
  say "Skipping the weight download (--no-models)"
fi

# ------------------------------------------------------------------ frontend
if [ "$BUILD_WEB" -eq 1 ]; then
  say "Building the web console"
  if ! command -v npm >/dev/null 2>&1; then
    warn "npm not found -- skipping the frontend. The API will still run."
  else
    ( cd web
      npm install --no-audit --no-fund
      npm run build )
  fi
else
  say "Skipping the frontend (--no-web)"
fi

# ------------------------------------------------------------------- verify
say "Verifying the GPU path"
"$PY" scripts/verify_gpu.py || warn "GPU verification reported problems (see above)"

say "Done"
cat <<'NEXT'
Next steps:

  1. Start the server:        .venv/bin/python -m server
  2. Open the console:        http://localhost:8000
  3. Replay a question:       .venv/bin/python tools/replay_device.py fixtures/q_math_7x23.wav
  4. Or just type into the console box -- same pipeline, no microphone.

Run the tests with:

  .venv/bin/python -m pytest -q      # 476 tests, ~40s
  cd web && npm test                 # 82 tests, ~1s
NEXT
