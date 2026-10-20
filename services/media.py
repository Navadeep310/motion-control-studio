from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2


def executable(name: str) -> str | None:
    return shutil.which(name)


def video_duration(path: Path) -> float:
    ffprobe = executable("ffprobe")
    if not ffprobe:
        raise RuntimeError(
            "ffprobe is missing. Install FFmpeg and make sure its bin folder is on PATH, "
            "then restart Motion Control Studio."
        )
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("The video could not be read. Check that it is a valid MP4 or MOV.")
    try:
        duration = float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Could not determine the video duration.") from exc
    if duration <= 0:
        raise ValueError("The video has no playable duration.")
    return duration


def sample_video_frames(path: Path, target_dir: Path, count: int = 5) -> list[Path]:
    ffmpeg = executable("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg is missing. Install FFmpeg and add its bin folder to PATH "
            "(Windows: winget install Gyan.FFmpeg; macOS: brew install ffmpeg; "
            "Ubuntu/Debian: sudo apt install ffmpeg), then restart the app."
        )
    duration = video_duration(path)
    target_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    for index in range(count):
        timestamp = duration * (index + 0.5) / count
        output = target_dir / f"frame_{index:02d}.jpg"
        result = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-vf",
                "scale=768:-2",
                "-y",
                str(output),
            ],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        if result.returncode != 0 or not output.is_file():
            raise ValueError("FFmpeg could not extract frames from the motion video.")
        frames.append(output)
    return frames


def combine_audio(video: Path, source: Path, destination: Path, keep_audio: bool) -> Path:
    ffmpeg = executable("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to finalize the generated video.")
    if keep_audio:
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            "-y",
            str(destination),
        ]
    else:
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-an",
            "-c:v",
            "copy",
            "-y",
            str(destination),
        ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    if result.returncode != 0:
        raise RuntimeError("FFmpeg could not finalize the result audio track.")
    return destination


def image_is_valid(path: Path) -> bool:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return image is not None and image.shape[0] > 0 and image.shape[1] > 0
