from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Callable

import httpx

from config import settings


Progress = Callable[[int, str], None]


def _workflow_for(
    image: Path,
    video: Path,
    prompt: str,
    work_dir: Path,
    orientation: str,
) -> dict:
    if not settings.workflow_path.is_file():
        raise RuntimeError(
            f"ComfyUI workflow not found at {settings.workflow_path}. Export a workflow in "
            "ComfyUI API format and save it there; see README.md."
        )
    try:
        workflow = json.loads(settings.workflow_path.read_text(encoding="utf-8"))
        image_node = workflow[settings.image_node_id]
        video_node = workflow[settings.video_node_id]
        prompt_node = workflow[settings.prompt_node_id]
        image_inputs = image_node["inputs"]
        video_inputs = video_node["inputs"]
        prompt_inputs = prompt_node["inputs"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(
            "The workflow JSON is invalid or its configured node IDs/inputs are missing. "
            "Set IMAGE_NODE_ID, VIDEO_NODE_ID, PROMPT_NODE_ID and input-name settings "
            "in .env to match your exported API workflow."
        ) from exc

    input_dir = settings.comfyui_input_dir
    if input_dir is None:
        candidates = [Path.home() / "ComfyUI" / "input"]
        input_dir = next((candidate for candidate in candidates if candidate.is_dir()), None)
    if input_dir is None or not input_dir.is_dir():
        raise RuntimeError(
            "ComfyUI's input folder is not configured. Set COMFYUI_INPUT_DIR in .env to "
            "the input directory used by this local ComfyUI installation, then restart."
        )
    image_copy = input_dir / f"motion-studio-{work_dir.name}{image.suffix.lower()}"
    video_copy = input_dir / f"motion-studio-{work_dir.name}{video.suffix.lower()}"
    try:
        shutil.copy2(image, image_copy)
        shutil.copy2(video, video_copy)
    except OSError as exc:
        raise RuntimeError(f"Cannot stage inputs in ComfyUI's input folder: {exc}") from exc
    image_inputs[settings.image_input_name] = image_copy.name
    video_inputs[settings.video_input_name] = video_copy.name
    orientation_guidance = (
        " Preserve the character's facing/orientation from the character image."
        if orientation == "image"
        else " Match the character's facing/orientation to the motion reference video."
    )
    prompt_inputs[settings.prompt_input_name] = prompt + orientation_guidance
    if settings.orientation_node_id:
        try:
            workflow[settings.orientation_node_id]["inputs"][settings.orientation_input_name] = (
                orientation
            )
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                "The configured orientation node or input is missing from the workflow. "
                "Check ORIENTATION_NODE_ID and ORIENTATION_INPUT_NAME in .env."
            ) from exc
    return workflow


def generate(
    image: Path,
    video: Path,
    prompt: str,
    work_dir: Path,
    orientation: str,
    progress: Progress,
    cancelled: Callable[[], bool],
) -> Path:
    workflow = _workflow_for(image, video, prompt, work_dir, orientation)
    client_id = str(uuid.uuid4())
    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                f"{settings.comfyui_url}/prompt",
                json={"prompt": workflow, "client_id": client_id},
            )
            response.raise_for_status()
            prompt_id = response.json()["prompt_id"]
            if not isinstance(prompt_id, str):
                raise ValueError("ComfyUI returned an invalid prompt ID.")
            deadline = time.monotonic() + settings.comfy_timeout_seconds
            while time.monotonic() < deadline:
                if cancelled():
                    try:
                        client.post(
                            f"{settings.comfyui_url}/interrupt",
                            json={"prompt_id": prompt_id},
                            timeout=5,
                        ).raise_for_status()
                    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                        raise RuntimeError(
                            f"Cancellation was requested, but ComfyUI could not confirm "
                            f"interruption: {exc}"
                        ) from exc
                    raise InterruptedError("Generation was canceled.")
                history_response = client.get(f"{settings.comfyui_url}/history/{prompt_id}")
                history_response.raise_for_status()
                history = history_response.json().get(prompt_id)
                if history:
                    status = history.get("status", {})
                    messages = json.dumps(status.get("messages", []), ensure_ascii=False).lower()
                    if any(term in messages for term in ("out of memory", "cuda error", "oom")):
                        raise RuntimeError(
                            "ComfyUI ran out of GPU memory. Reduce the workflow resolution, "
                            "frame count, or batch size, then try again."
                        )
                    if status.get("status_str") == "error" or status.get("completed") is False:
                        raise RuntimeError("ComfyUI could not complete the workflow. Check its console log.")
                    output = _find_video_output(history.get("outputs", {}))
                    if output:
                        filename, subfolder, file_type = output
                        result = client.get(
                            f"{settings.comfyui_url}/view",
                            params={"filename": filename, "subfolder": subfolder, "type": file_type},
                        )
                        result.raise_for_status()
                        destination = work_dir / "generated.mp4"
                        destination.write_bytes(result.content)
                        if not destination.stat().st_size:
                            raise RuntimeError("ComfyUI returned an empty video file.")
                        progress(92, "Video generated; finalizing audio")
                        return destination
                progress(20, "Waiting for ComfyUI generation")
                time.sleep(2)
            raise RuntimeError(
                f"ComfyUI timed out after {settings.comfy_timeout_seconds} seconds. "
                "Check the ComfyUI console and try a smaller workflow."
            )
    except InterruptedError:
        raise
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"Cannot connect to ComfyUI at {settings.comfyui_url}. Start ComfyUI and check "
            "COMFYUI_URL in .env."
        ) from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError("ComfyUI request timed out. Check the ComfyUI console and try again.") from exc
    except httpx.HTTPStatusError as exc:
        body = exc.response.text.lower()
        if "out of memory" in body or "cuda" in body and "memory" in body:
            raise RuntimeError(
                "ComfyUI ran out of GPU memory. Reduce workflow resolution, frame count, or batch size."
            ) from exc
        raise RuntimeError(f"ComfyUI returned HTTP {exc.response.status_code}.") from exc
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"ComfyUI returned an invalid response: {exc}") from exc


def _find_video_output(outputs: dict) -> tuple[str, str, str] | None:
    for node_output in outputs.values():
        for key in ("videos", "gifs", "images"):
            for item in node_output.get(key, []):
                filename = item.get("filename", "")
                if Path(filename).suffix.lower() == ".mp4":
                    return filename, item.get("subfolder", ""), item.get("type", "output")
    return None
