#!/usr/bin/env bash
# =============================================================================
# run.sh — Fully automated GRPO Drone2D training pipeline
#
# Tested environment:
#   Ubuntu 22.04, NVIDIA GPU (CUDA 12.x), Python 3.12
#   All package versions are pinned to the exact versions used in development.
#
# What this script does:
#   1. Installs system dependencies (Python 3.12, build tools, GLFW, etc.)
#   2. Installs uv (fast Python package manager)
#   3. Creates a Python 3.12 virtual environment
#   4. Installs PyTorch 2.11.0 with CUDA support
#   5. Installs all project dependencies at exact pinned versions
#   6. Installs vLLM 0.20.1 (kept outside pyproject.toml due to numpy conflict)
#   7. Fixes the libnvJitLink.so.13 symlink required by bitsandbytes
#   8. Runs GRPO training  (TRAIN_EPOCHS / SAMPLES_PER_ENV control duration)
#   9. Generates static training-metric plots (no display required)
#  10. Runs the full stats checker on the training results
#  11. Copies all outputs to ./results/
#
# Duration:
#   Default (TRAIN_EPOCHS=1, SAMPLES_PER_ENV=5):  ~15–25 min on an RTX 3080
#   Full run (TRAIN_EPOCHS=4, SAMPLES_PER_ENV=25): ~2–3 h
#
# Override example:
#   TRAIN_EPOCHS=2 SAMPLES_PER_ENV=10 bash run.sh
# =============================================================================

set -euo pipefail

# ── Resolve absolute project root regardless of where run.sh is called from ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RESULTS_DIR="$SCRIPT_DIR/results"
METRICS_FILE="$SCRIPT_DIR/checkpoints/drone_grpo/metrics.jsonl"
LOG_FILE="$RESULTS_DIR/run.log"

mkdir -p "$RESULTS_DIR"

# Redirect all output to log file AND terminal
exec > >(tee -a "$LOG_FILE") 2>&1

echo ""
echo "============================================================"
echo "  GRPO Drone2D — Automated Training Pipeline"
echo "  $(date)"
echo "  Working directory: $SCRIPT_DIR"
echo "============================================================"
echo ""

# ── Training duration knobs ───────────────────────────────────────────────────
# Set via environment variables before calling this script.
# Defaults give a fast demo run (~15 min):
#   TRAIN_EPOCHS=1       →  1 full pass through the dataset
#   SAMPLES_PER_ENV=5    →  5 × 16 envs = 80 prompts → 80 steps at ~4 s each
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"
export SAMPLES_PER_ENV="${SAMPLES_PER_ENV:-5}"

echo "[config] TRAIN_EPOCHS=${TRAIN_EPOCHS}  SAMPLES_PER_ENV=${SAMPLES_PER_ENV}"
echo ""

# =============================================================================
# STEP 1 — System dependencies
# =============================================================================
echo "==> [1/7] Installing system dependencies..."

apt-get update -qq 2>&1 | tail -1

apt-get install -y -qq \
    software-properties-common \
    curl \
    wget \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libxrender1 \
    libxext6 \
    libx11-6 \
    libglfw3 \
    libglfw3-dev \
    pkg-config \
    2>&1 | tail -5

# Python 3.12 (ubuntu:22.04 ships 3.10 by default)
if ! python3.12 --version &>/dev/null 2>&1; then
    echo "  Installing Python 3.12 via deadsnakes PPA..."
    add-apt-repository -y ppa:deadsnakes/ppa -q
    apt-get update -qq
    apt-get install -y -qq python3.12 python3.12-venv python3.12-dev python3.12-distutils
fi

echo "  Python: $(python3.12 --version)"
echo "  System deps OK."
echo ""

# =============================================================================
# STEP 2 — Install uv
# =============================================================================
echo "==> [2/7] Installing uv package manager..."

if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# uv installs to ~/.cargo/bin or ~/.local/bin depending on the install method
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

if ! command -v uv &>/dev/null; then
    echo "  ERROR: uv not found after installation. Trying pip install uv..."
    python3.12 -m pip install uv --quiet
fi

echo "  uv: $(uv --version)"
echo ""

# =============================================================================
# STEP 3 — Create virtual environment
# =============================================================================
echo "==> [3/7] Creating Python 3.12 virtual environment..."

if [ ! -d ".venv" ]; then
    uv venv --python python3.12 .venv
    echo "  Created .venv"
else
    echo "  .venv already exists — reusing"
fi

# Activate for the rest of the script
source .venv/bin/activate
export PYTHONPATH="$SCRIPT_DIR"

echo "  Python: $(python --version)"
echo ""

# =============================================================================
# STEP 4 — Install PyTorch 2.11.0 (CUDA)
# =============================================================================
echo "==> [4/7] Installing PyTorch 2.11.0..."

# Detect CUDA version to pick the correct wheel index.
# Supported: cu118 (CUDA 11.8), cu121 (CUDA 12.1), cu124 (CUDA 12.4)
detect_cuda_index() {
    local ver
    ver=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1 || echo "")
    if [ -z "$ver" ]; then
        # Fallback: check nvidia-smi
        ver=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1 || echo "")
    fi
    local major
    major=$(echo "$ver" | cut -d. -f1)
    local minor
    minor=$(echo "$ver" | cut -d. -f2)
    if [ -z "$major" ]; then
        echo "cpu"
    elif [ "$major" -ge 12 ] && [ "$minor" -ge 4 ]; then
        echo "cu124"
    elif [ "$major" -ge 12 ] && [ "$minor" -ge 1 ]; then
        echo "cu121"
    else
        echo "cu118"
    fi
}

CUDA_INDEX=$(detect_cuda_index)
echo "  Detected CUDA index: $CUDA_INDEX"

if [ "$CUDA_INDEX" = "cpu" ]; then
    echo "  WARNING: No GPU detected. Training will run on CPU (very slow)."
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
else
    TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_INDEX}"
fi

# Install PyTorch first (separate step — needs special index URL)
uv pip install \
    "torch==2.11.0" \
    "torchvision" \
    --index-url "$TORCH_INDEX" \
    --quiet

echo "  PyTorch $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# =============================================================================
# STEP 5 — Install project dependencies (exact pinned versions)
# =============================================================================
echo "==> [5/7] Installing project dependencies..."

# Install from pyproject.toml (uv respects the uv.lock if present)
uv pip install -e . --quiet

# Install exact versions used and validated during development
uv pip install \
    "accelerate==1.13.0" \
    "bitsandbytes==0.49.2" \
    "datasets==4.8.4" \
    "matplotlib==3.10.9" \
    "numpy==2.4.4" \
    "peft==0.19.1" \
    "pydantic>=2.0.0" \
    "pybullet==3.2.7" \
    "transformers==5.5.4" \
    "trl==1.1.0" \
    "httpx>=0.28.0" \
    "glfw>=2.10.0" \
    "moderngl>=5.12.0" \
    "moderngl-window>=3.1.1" \
    --quiet

echo "  Core deps OK."

# vLLM 0.20.1 is installed AFTER other deps because it has a numpy<2.4
# transitive constraint from mistral-common[image] that would downgrade numpy.
# Installing it last lets uv satisfy its numpy constraint separately.
echo "  Installing vLLM 0.20.1 (this may take a few minutes)..."
uv pip install "vllm==0.20.1" --quiet

echo "  vLLM $(python -c 'import vllm; print(vllm.__version__)')"
echo ""

# =============================================================================
# STEP 6 — Fix libnvJitLink symlink (required by bitsandbytes on CUDA 12.x)
# =============================================================================
echo "==> [6/7] Checking libnvJitLink symlink..."

fix_nvjitlink() {
    # Find any libnvJitLink.so.1* on the system
    local src
    src=$(find /usr/local/cuda /usr/lib -name "libnvJitLink.so.1*" 2>/dev/null | sort | tail -1 || echo "")
    if [ -z "$src" ]; then
        echo "  libnvJitLink not found — skipping (may be fine if bitsandbytes not used)"
        return
    fi
    local libdir
    libdir=$(dirname "$src")
    local target="$libdir/libnvJitLink.so.13"
    if [ ! -f "$target" ] && [ ! -L "$target" ]; then
        echo "  Creating symlink: $target → $src"
        ln -sf "$src" "$target" 2>/dev/null || true
        ldconfig 2>/dev/null || true
    else
        echo "  libnvJitLink.so.13 already present — OK"
    fi
}
fix_nvjitlink
echo ""

# =============================================================================
# STEP 7 — Run GRPO training
# =============================================================================
echo "==> [7/10] Starting GRPO training..."
echo "  TRAIN_EPOCHS=${TRAIN_EPOCHS}  SAMPLES_PER_ENV=${SAMPLES_PER_ENV}"
echo "  Expected steps: $((SAMPLES_PER_ENV * 16)) per epoch × ${TRAIN_EPOCHS} epoch(s)"
echo "  Model will be saved to: ./checkpoints/drone_grpo/final/"
echo ""

TRAIN_START=$(date +%s)

TRAIN_EPOCHS="$TRAIN_EPOCHS" \
SAMPLES_PER_ENV="$SAMPLES_PER_ENV" \
PYTHONPATH="$SCRIPT_DIR" \
python training/train.py

TRAIN_END=$(date +%s)
TRAIN_SECS=$((TRAIN_END - TRAIN_START))
echo ""
echo "  Training finished in $((TRAIN_SECS / 60))m $((TRAIN_SECS % 60))s"
echo ""

# =============================================================================
# STEP 8 — Generate static plots
# =============================================================================
echo "==> [8/10] Generating training plots..."

if [ ! -f "$METRICS_FILE" ]; then
    echo "  WARNING: metrics file not found at $METRICS_FILE — skipping plots"
else
    MPLBACKEND=Agg \
    PYTHONPATH="$SCRIPT_DIR" \
    python training/generate_static_plots.py \
        --metrics "$METRICS_FILE" \
        --out     "$RESULTS_DIR"
fi
echo ""

# =============================================================================
# STEP 9 — Run stats checker
# =============================================================================
echo "==> [9/10] Running stats checker..."

if [ ! -f "$METRICS_FILE" ]; then
    echo "  WARNING: metrics file not found — skipping stats"
else
    PYTHONPATH="$SCRIPT_DIR" \
    python stats_checker.py 2>&1 | tee "$RESULTS_DIR/stats_checker_output.txt"
fi
echo ""

# =============================================================================
# STEP 10 — Collect all outputs into results/
# =============================================================================
echo "==> [10/10] Collecting outputs..."

# Copy metrics log
[ -f "$METRICS_FILE" ] && cp "$METRICS_FILE" "$RESULTS_DIR/metrics.jsonl"

# Copy iteration timer stats if present
ITER_STATS="$SCRIPT_DIR/checkpoints/drone_grpo/metrics.jsonl"
if [ -f "$ITER_STATS" ]; then
    PYTHONPATH="$SCRIPT_DIR" \
    python check_iteration_timer_stats.py 2>/dev/null \
        > "$RESULTS_DIR/iteration_timing.txt" || true
fi

# Copy saved model card if present
MODEL_README="$SCRIPT_DIR/checkpoints/drone_grpo/README.md"
[ -f "$MODEL_README" ] && cp "$MODEL_README" "$RESULTS_DIR/model_card.md"

echo ""
echo "============================================================"
echo "  Pipeline complete!"
echo "  $(date)"
echo ""
echo "  Outputs in: $RESULTS_DIR/"
ls -lh "$RESULTS_DIR/" 2>/dev/null || true
echo ""
echo "  Key files:"
echo "    results/training_dashboard.png  — 9-panel training metrics chart"
echo "    results/reward_analysis.png     — reward distribution and trajectory"
echo "    results/training_summary.txt    — human-readable stats summary"
echo "    results/stats_checker_output.txt — full stats_checker.py output"
echo "    results/metrics.jsonl           — raw per-step training metrics"
echo "    results/run.log                 — full pipeline log"
echo ""
echo "  Trained model: checkpoints/drone_grpo/final/"
echo "============================================================"
