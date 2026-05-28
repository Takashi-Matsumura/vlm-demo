import asyncio
import os
import tempfile
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from inference import get_status, is_ready, load_model, run_stream


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Blocks until model is fully loaded. For 72B this takes 1–3 min on M3 Ultra.
    load_model()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/v1/health")
async def health():
    return get_status()


@app.post("/v1/infer")
async def infer(
    prompt: str = Form(...),
    video: UploadFile | None = File(default=None),
    image: UploadFile | None = File(default=None),
    max_tokens: int = Form(default=1024),
):
    if not is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    if video is None and image is None:
        raise HTTPException(status_code=400, detail="`video` or `image` field is required")

    upload = video if video is not None else image
    raw_name = upload.filename or "upload"
    suffix = os.path.splitext(raw_name)[1] or (".mp4" if video is not None else ".jpg")

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        contents = await upload.read()
        tmp.write(contents)
        tmp.flush()
        tmp.close()
        tmp_path = tmp.name
    except Exception:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise

    video_path = tmp_path if video is not None else None
    image_path = tmp_path if image is not None else None

    # MLX Metal stream is thread-local; run generation in the main thread to avoid
    # "There is no Stream(gpu, 1) in current thread" errors from run_in_executor.
    async def generate_chunks() -> AsyncGenerator[bytes, None]:
        try:
            for text in run_stream(prompt, video_path, image_path, max_tokens):
                yield text.encode("utf-8")
                await asyncio.sleep(0)  # yield event loop between tokens
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return StreamingResponse(generate_chunks(), media_type="text/plain; charset=utf-8")
