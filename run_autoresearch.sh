#!/bin/bash
# Run autoresearch with proper GPU VRAM management.
# Uses systemctl mask/stop to prevent Ollama auto-restart during training.
# Usage: ./run_autoresearch.sh [--max-iter N] [--model MODEL]

set -e

# Activate lora-env
source ~/ai/lora-env/bin/activate
cd ~/ai/autoquant

# Parse args (pass through to local_autoresearch.py)
ARGS="$@"
if [ -z "$ARGS" ]; then
    ARGS="--max-iter 3"
fi

echo "=== Autoresearch wrapper ==="
echo "Args: $ARGS"
echo "Time: $(date)"
echo ""

# Run autoresearch
python3 local_autoresearch.py $ARGS 2>&1

# Ensure Ollama is running when we're done (even on error)
echo ""
echo "=== Ensuring Ollama is restored ==="
systemctl unmask ollama 2>/dev/null || true
systemctl start ollama 2>/dev/null || true
echo "Done at $(date)"
