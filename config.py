from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


@dataclass
class Settings:
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    vision_model: str = os.getenv("VISION_MODEL", "qwen2.5vl")
    comfyui_url: str = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
    comfyui_input_dir: Path | None = (
        Path(os.environ["COMFYUI_INPUT_DIR"]).expanduser()
        if os.getenv("COMFYUI_INPUT_DIR")
        else None
    )
    workflow_path: Path = Path(
        os.getenv("WORKFLOW_PATH", str(ROOT / "workflows" / "wan_animate_api.json"))
    )
    image_node_id: str = os.getenv("IMAGE_NODE_ID", "10")
    video_node_id: str = os.getenv("VIDEO_NODE_ID", "11")
    prompt_node_id: str = os.getenv("PROMPT_NODE_ID", "12")
    image_input_name: str = os.getenv("IMAGE_INPUT_NAME", "image")
    video_input_name: str = os.getenv("VIDEO_INPUT_NAME", "video")
    prompt_input_name: str = os.getenv("PROMPT_INPUT_NAME", "text")
    vision_timeout_seconds: int = int(os.getenv("VISION_TIMEOUT_SECONDS", "600"))
    orientation_node_id: str = os.getenv("ORIENTATION_NODE_ID", "")
    orientation_input_name: str = os.getenv("ORIENTATION_INPUT_NAME", "orientation")
    comfy_timeout_seconds: int = int(os.getenv("COMFY_TIMEOUT_SECONDS", "600"))
    max_video_bytes: int = 50 * 1024 * 1024
    max_video_seconds: float = 10
    data_dir: Path = Path(os.getenv("DATA_DIR", str(ROOT / "data"))).resolve()


settings = Settings()
