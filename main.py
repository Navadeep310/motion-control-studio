from __future__ import annotations

import logging
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import ROOT, settings
from services import comfy, lite, media, ollama


app = FastAPI(
    title="Motion Control Studio",
    description="A local, open-source approximation for character motion transfer.",
    version="1.0.0",
)
logger = logging.getLogger(__name__)


@dataclass
class Prepared:
    directory: Path
    image: Path
    video: Path
    keep_audio: bool
    orientation: str
    vision: dict[str, str | bool]


@dataclass
class Job:
    id: str
    prepared_id: str
    backend: str
    prompt: str
    state: str = "queued"
    progress: int = 0
    message: str = "Waiting to start"
    error: str | None = None
    result: Path | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)


prepared_items: dict[str, Prepared] = {}
jobs: dict[str, Job] = {}
state_lock = threading.RLock()


class GenerateRequest(BaseModel):
    prepared_id: str
    backend: str = Field(pattern="^(full|lite)$")
    prompt: str = Field(min_length=1, max_length=2000)


def _save_upload(upload: UploadFile, path: Path, max_bytes: int) -> None:
    total = 0
    try:
        with path.open("wb") as output:
            while chunk := upload.file.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail="The uploaded file exceeds the size limit.")
                output.write(chunk)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not save the upload: {exc}") from exc
    if total == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")


def _progress(job_id: str, percent: int, message: str) -> None:
    with state_lock:
        job = jobs.get(job_id)
        if job and job.state == "running":
            job.progress = max(job.progress, min(99, percent))
            job.message = message


def _run_job(job_id: str) -> None:
    with state_lock:
        job = jobs[job_id]
        if job.cancel_event.is_set():
            job.state = "canceled"
            job.message = "Generation canceled"
            return
        prepared = prepared_items[job.prepared_id]
        job.state = "running"
        job.progress = 5
        job.message = "Starting generation"
    work_dir = prepared.directory
    generated = work_dir / "generated.mp4"
    final = work_dir / "result.mp4"
    try:
        if job.cancel_event.is_set():
            raise InterruptedError("Generation was canceled.")
        if job.backend == "full":
            generated = comfy.generate(
                prepared.image,
                prepared.video,
                job.prompt,
                work_dir,
                prepared.orientation,
                lambda percent, message: _progress(job_id, percent, message),
                job.cancel_event.is_set,
            )
        else:
            generated = lite.generate(
                prepared.image,
                prepared.video,
                work_dir / "lite_no_audio.mp4",
                lambda percent, message: _progress(job_id, percent, message),
                job.cancel_event.is_set,
            )
        if job.cancel_event.is_set():
            raise InterruptedError("Generation was canceled.")
        media.combine_audio(generated, prepared.video, final, prepared.keep_audio)
        with state_lock:
            job.result = final
            job.state = "completed"
            job.progress = 100
            job.message = "Generation complete"
    except InterruptedError:
        with state_lock:
            job.state = "canceled"
            job.message = "Generation canceled"
    except (RuntimeError, ValueError) as exc:
        with state_lock:
            job.state = "failed"
            job.error = str(exc)
            job.message = "Generation failed"
    except Exception:
        with state_lock:
            job.state = "failed"
            job.error = "An unexpected error occurred. Check the server console for details."
            job.message = "Generation failed"
        logger.exception("Unexpected generation error for job %s", job_id)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health() -> dict:
    ollama_status = "ok"
    ollama_message = f"Connected to {settings.ollama_url}"
    models: list[str] = []
    try:
        response = httpx.get(f"{settings.ollama_url}/api/tags", timeout=3)
        response.raise_for_status()
        models = [item.get("name", "") for item in response.json().get("models", [])]
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError):
        ollama_status = "missing"
        ollama_message = (
            f"Ollama is not responding at {settings.ollama_url}. Start it with `ollama serve`."
        )
    vision_available = any(
        model.split(":")[0] == settings.vision_model.split(":")[0] for model in models
    )
    vision_status = "ok" if vision_available else "missing"
    vision_message = (
        f"{settings.vision_model} is available"
        if vision_available
        else f"Vision model {settings.vision_model} is missing. Install it with "
        f"`ollama pull {settings.vision_model}`."
    )

    ffmpeg_status = "ok" if media.executable("ffmpeg") else "missing"
    ffmpeg_message = (
        "ffmpeg and ffprobe are available"
        if ffmpeg_status == "ok" and media.executable("ffprobe")
        else "FFmpeg tools are missing or incomplete. Install FFmpeg and add its bin folder to PATH "
        "(Windows: `winget install Gyan.FFmpeg`; macOS: `brew install ffmpeg`; "
        "Ubuntu/Debian: `sudo apt install ffmpeg`), then restart the app."
    )
    if not media.executable("ffprobe"):
        ffmpeg_status = "missing"

    comfy_status = "ok"
    comfy_message = f"Connected to {settings.comfyui_url}"
    try:
        response = httpx.get(f"{settings.comfyui_url}/system_stats", timeout=3)
        response.raise_for_status()
    except (httpx.RequestError, httpx.HTTPStatusError):
        comfy_status = "missing"
        comfy_message = (
            f"ComfyUI is not responding at {settings.comfyui_url}. Start ComfyUI; "
            "Lite mode can still be used without it."
        )
    workflow_ok = False
    try:
        import json

        workflow = json.loads(settings.workflow_path.read_text(encoding="utf-8"))
        workflow_ok = all(
            node_id in workflow
            and isinstance(workflow[node_id].get("inputs"), dict)
            and input_name in workflow[node_id]["inputs"]
            for node_id, input_name in (
                (settings.image_node_id, settings.image_input_name),
                (settings.video_node_id, settings.video_input_name),
                (settings.prompt_node_id, settings.prompt_input_name),
            )
        )
    except (OSError, ValueError, TypeError, AttributeError):
        workflow_ok = False
    workflow_message = (
        f"API workflow is configured at {settings.workflow_path}"
        if workflow_ok
        else f"Wan Animate API workflow is missing or its configured nodes are invalid at "
        f"{settings.workflow_path}. Export an API workflow and set node IDs/inputs in .env; "
        "see README.md."
    )
    return {
        "checks": [
            {"name": "Ollama", "status": ollama_status, "message": ollama_message},
            {"name": "Vision model", "status": vision_status, "message": vision_message},
            {"name": "FFmpeg", "status": ffmpeg_status, "message": ffmpeg_message},
            {"name": "ComfyUI", "status": comfy_status, "message": comfy_message},
            {
                "name": "Wan workflow",
                "status": "ok" if workflow_ok else "missing",
                "message": workflow_message,
            },
        ]
    }


@app.post("/api/prepare")
def prepare(
    character: UploadFile = File(...),
    motion: UploadFile = File(...),
    orientation: str = Form("image"),
    keep_audio: bool = Form(True),
) -> dict:
    if orientation not in {"image", "video"}:
        raise HTTPException(status_code=400, detail="Orientation must be 'image' or 'video'.")
    image_ext = Path(character.filename or "").suffix.lower()
    video_ext = Path(motion.filename or "").suffix.lower()
    if image_ext not in {".png", ".jpg", ".jpeg"}:
        raise HTTPException(status_code=415, detail="Character image must be a PNG or JPG file.")
    if video_ext not in {".mp4", ".mov"}:
        raise HTTPException(status_code=415, detail="Motion video must be an MP4 or MOV file.")

    identifier = str(uuid.uuid4())
    directory = settings.data_dir / "uploads" / identifier
    directory.mkdir(parents=True, exist_ok=True)
    image_path = directory / f"character{image_ext}"
    video_path = directory / f"motion{video_ext}"
    try:
        _save_upload(character, image_path, 20 * 1024 * 1024)
        _save_upload(motion, video_path, settings.max_video_bytes)
    except HTTPException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    if not media.image_is_valid(image_path):
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(status_code=400, detail="The character image file is invalid or unreadable.")
    try:
        duration = media.video_duration(video_path)
        if duration > settings.max_video_seconds:
            raise HTTPException(
                status_code=413,
                detail=f"The motion video is {duration:.1f} seconds long; the maximum is 10 seconds.",
            )
        frames = media.sample_video_frames(video_path, directory / "frames", count=5)
        vision = ollama.inspect_and_prompt(image_path, frames)
    except HTTPException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    except (RuntimeError, ValueError) as exc:
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Could not process the uploads: {exc}") from exc
    with state_lock:
        prepared_items[identifier] = Prepared(
            directory, image_path, video_path, keep_audio, orientation, vision
        )
    return {"prepared_id": identifier, **vision}


@app.post("/api/generate", status_code=202)
def generate(request: GenerateRequest, background_tasks: BackgroundTasks) -> dict:
    with state_lock:
        if request.prepared_id not in prepared_items:
            raise HTTPException(status_code=404, detail="Prepared upload not found. Upload the files again.")
        if request.backend == "full" and not settings.workflow_path.is_file():
            raise HTTPException(
                status_code=400,
                detail=f"ComfyUI workflow not found at {settings.workflow_path}. "
                "Export the API workflow as described in README.md.",
            )
        identifier = str(uuid.uuid4())
        job = Job(identifier, request.prepared_id, request.backend, request.prompt.strip())
        jobs[identifier] = job
    background_tasks.add_task(_run_job, identifier)
    return {"job_id": identifier, "state": "queued"}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    with state_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Generation job not found.")
        result = {
            "job_id": job.id,
            "state": job.state,
            "progress": job.progress,
            "message": job.message,
            "error": job.error,
        }
        if job.state == "completed":
            result["download_url"] = f"/api/jobs/{job.id}/download"
            result["video_url"] = f"/api/jobs/{job.id}/video"
        return result


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    with state_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Generation job not found.")
        if job.state in {"completed", "failed", "canceled"}:
            return {"job_id": job.id, "state": job.state, "message": "Job has already finished."}
        job.cancel_event.set()
        if job.state == "queued":
            job.state = "canceled"
            job.message = "Generation canceled"
        return {"job_id": job.id, "state": job.state, "message": "Cancellation requested."}


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str) -> FileResponse:
    with state_lock:
        job = jobs.get(job_id)
        if job is None or job.state != "completed" or job.result is None:
            raise HTTPException(status_code=404, detail="The result video is not ready.")
        result = job.result
    return FileResponse(result, media_type="video/mp4", filename="motion-control-result.mp4")


@app.get("/api/jobs/{job_id}/video")
def play_video(job_id: str) -> FileResponse:
    with state_lock:
        job = jobs.get(job_id)
        if job is None or job.state != "completed" or job.result is None:
            raise HTTPException(status_code=404, detail="The result video is not ready.")
        result = job.result
    return FileResponse(
        result,
        media_type="video/mp4",
        filename="motion-control-result.mp4",
        content_disposition_type="inline",
    )
