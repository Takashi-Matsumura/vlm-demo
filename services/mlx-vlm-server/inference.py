import os
from typing import Generator

from mlx_vlm import load, stream_generate

MODEL_ID = os.environ.get("MLX_MODEL_ID", "mlx-community/Qwen2.5-VL-72B-Instruct-bf16")

_model = None
_processor = None
_load_error: str | None = None


def load_model() -> None:
    """Load model into memory once at startup. Blocks for 1–15 min depending on model size."""
    global _model, _processor, _load_error
    try:
        _model, _processor = load(MODEL_ID)
        _load_error = None
    except Exception as exc:
        _load_error = str(exc)
        raise


def is_ready() -> bool:
    return _model is not None and _load_error is None


def get_status() -> dict:
    return {"model_id": MODEL_ID, "loaded": is_ready(), "error": _load_error}


def _build_prompt(user_text: str, video_path: str | None, image_path: str | None) -> str:
    """Format prompt using processor's chat template with proper media tokens."""
    if video_path:
        content = [
            {"type": "video", "video": video_path, "fps": 2.0},
            {"type": "text", "text": user_text},
        ]
    else:
        content = [
            {"type": "image", "image": image_path},
            {"type": "text", "text": user_text},
        ]

    messages = [{"role": "user", "content": content}]

    # Use processor's own chat_template (Qwen2.5-VL knows about video tokens)
    processor = _processor
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    chat_template = getattr(processor, "chat_template", None) or getattr(tokenizer, "chat_template", None)

    if chat_template:
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # Fallback: plain text prompt
    return user_text


def run_stream(
    prompt: str,
    video_path: str | None,
    image_path: str | None,
    max_tokens: int = 1024,
) -> Generator[str, None, None]:
    """Synchronous blocking generator — yields text chunks as they are produced."""
    if not is_ready():
        raise RuntimeError("Model not loaded")

    formatted = _build_prompt(prompt, video_path, image_path)

    for chunk in stream_generate(
        _model,
        _processor,
        formatted,
        image=image_path,
        video=video_path,
        max_tokens=max_tokens,
        repetition_penalty=1.15,
        repetition_context_size=20,
    ):
        if chunk.text:
            yield chunk.text
