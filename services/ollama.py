from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import httpx
import numpy as np

from config import settings


def _image_base64(path: Path) -> str:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image for vision analysis: {path.name}")
    height, width = image.shape[:2]
    longest_edge = max(height, width)
    if longest_edge > 640:
        scale = 640 / longest_edge
        image = cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 82],
    )
    if not ok:
        raise RuntimeError(f"Could not encode image for vision analysis: {path.name}")
    return base64.b64encode(np.asarray(encoded).tobytes()).decode("ascii")


def inspect_and_prompt(character: Path, frames: list[Path]) -> dict[str, str | bool]:
    instruction = (
        "You are the vision and prompt-planning component of a local animation tool. "
        "Inspect image 1 as the character image and all remaining images as ordered "
        "sampled frames from the motion reference video. Do not claim to generate video. "
        "Return JSON only with these fields: person_visible (boolean; true only if a "
        "recognizable person or character is visible in image 1), "
        "character_description (one concise sentence), "
        "motion_description (one concise sentence about the reference motion), "
        "generation_prompt (a short usable video-generation prompt mentioning character "
        "appearance, motion style, and camera). If image 1 does not show a person or "
        "character, set person_visible to false."
    )
    images = [_image_base64(character), *(_image_base64(frame) for frame in frames)]
    try:
        response = httpx.post(
            f"{settings.ollama_url}/api/generate",
            json={
                "model": settings.vision_model,
                "prompt": instruction,
                "images": images,
                "stream": False,
                "format": "json",
                "keep_alive": "10m",
                "options": {"num_predict": 200},
            },
            timeout=settings.vision_timeout_seconds,
        )
        response.raise_for_status()
        raw = response.json().get("response", "")
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"Cannot connect to Ollama at {settings.ollama_url}. Start Ollama with "
            "`ollama serve` and verify that it is listening."
        ) from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError("Ollama timed out while examining the images. Try again.") from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Ollama returned HTTP {exc.response.status_code}.") from exc
    except (ValueError, httpx.RequestError) as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    try:
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError("response is not an object")
        visible = result["person_visible"]
        description = result["character_description"]
        motion = result["motion_description"]
        prompt = result["generation_prompt"]
        if (
            not isinstance(visible, bool)
            or not all(isinstance(item, str) and item.strip() for item in (description, motion, prompt))
        ):
            raise ValueError("response fields have invalid types or empty values")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Ollama returned an invalid vision response. Check that {settings.vision_model} "
            "is a compatible vision model and try again."
        ) from exc
    if not visible:
        raise ValueError(
            "The vision model could not find a visible person or character in the uploaded "
            "image. Choose an image with a clearly visible full or upper-body character."
        )
    return {
        "person_visible": visible,
        "character_description": description.strip(),
        "motion_description": motion.strip(),
        "generation_prompt": prompt.strip(),
    }
