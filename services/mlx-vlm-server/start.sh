#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

source .venv/bin/activate

MLX_MODEL_ID="${MLX_MODEL_ID:-mlx-community/Qwen2.5-VL-7B-Instruct-8bit}"
PORT="${PORT:-8001}"

echo "Starting mlx-vlm-server with model: $MLX_MODEL_ID on port $PORT"
MLX_MODEL_ID="$MLX_MODEL_ID" uvicorn main:app --host 127.0.0.1 --port "$PORT" --timeout-keep-alive 600
