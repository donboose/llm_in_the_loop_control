#!/usr/bin/env bash
# Run GRPO training, auto-restarting on crash up to MAX_RETRIES times.
# Auto-resume logic inside train.py picks up from the latest checkpoint.
#
# Usage:
#   chmod +x train_overnight.sh
#   ./train_overnight.sh          # run in foreground
#   nohup ./train_overnight.sh &  # run detached (output → nohup.out)

set -euo pipefail

MAX_RETRIES=10
RETRY_DELAY=30   # seconds to wait between retries
ATTEMPT=0

cd "$(dirname "$0")"

while [ $ATTEMPT -lt $MAX_RETRIES ]; do
    ATTEMPT=$((ATTEMPT + 1))
    echo ""
    echo "------------------------------------------------"
    echo "  Training attempt $ATTEMPT / $MAX_RETRIES"
    echo "------------------------------------------------"

    # --resume: pick up from latest checkpoint if one exists (safe on first run too —
    if PYTHONPATH=. uv run python training/train.py --resume; then
        echo ""
        echo "Training completed successfully on attempt $ATTEMPT."
        exit 0
    fi

    EXIT_CODE=$?
    echo ""
    echo "[overnight] Training exited with code $EXIT_CODE on attempt $ATTEMPT."

    if [ $ATTEMPT -lt $MAX_RETRIES ]; then
        echo "[overnight] Waiting ${RETRY_DELAY}s before retry..."
        sleep $RETRY_DELAY
    fi
done

echo ""
echo "[overnight] Reached max retries ($MAX_RETRIES). Check the logs."
exit 1
