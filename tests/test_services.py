from __future__ import annotations

import json

import pytest

from config import settings
from services import comfy, lite, ollama
from tests.fakes import FakeComfyHandler, FakeOllamaHandler, FakeServer, sample_image_bytes


@pytest.fixture
def vision_server(monkeypatch):
    FakeOllamaHandler.result = {
        "person_visible": True,
        "character_description": "A red-jacketed character.",
        "motion_description": "The subject raises both arms.",
        "generation_prompt": "A red-jacketed character raises both arms, steady camera.",
    }
    FakeOllamaHandler.request_json = None
    with FakeServer(FakeOllamaHandler) as url:
        monkeypatch.setattr(settings, "ollama_url", url)
        yield


def test_fake_ollama_reads_character_and_five_video_frames(tmp_path, vision_server):
    character = tmp_path / "character.png"
    character.write_bytes(sample_image_bytes(".png"))
    frames = []
    for index in range(5):
        frame = tmp_path / f"frame-{index}.jpg"
        frame.write_bytes(sample_image_bytes(".jpg"))
        frames.append(frame)

    result = ollama.inspect_and_prompt(character, frames)

    assert result["person_visible"] is True
    assert result["generation_prompt"].startswith("A red-jacketed")
    sent = FakeOllamaHandler.request_json
    assert sent is not None
    assert sent["model"] == settings.vision_model
    assert len(sent["images"]) == 6
    assert sent["stream"] is False


def test_fake_ollama_rejects_image_without_visible_person(tmp_path, monkeypatch):
    FakeOllamaHandler.result = {
        "person_visible": False,
        "character_description": "An empty room.",
        "motion_description": "A person walks.",
        "generation_prompt": "A person walks.",
    }
    character = tmp_path / "character.png"
    character.write_bytes(sample_image_bytes(".png"))
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(sample_image_bytes(".jpg"))
    with FakeServer(FakeOllamaHandler) as url:
        monkeypatch.setattr(settings, "ollama_url", url)
        with pytest.raises(ValueError, match="could not find a visible person"):
            ollama.inspect_and_prompt(character, [frame])


@pytest.fixture
def comfy_server(monkeypatch):
    FakeComfyHandler.prompt_body = None
    FakeComfyHandler.mode = "success"
    with FakeServer(FakeComfyHandler) as url:
        monkeypatch.setattr(settings, "comfyui_url", url)
        yield


def _configure_workflow(tmp_path, monkeypatch):
    workflow_path = tmp_path / "workflow.json"
    workflow_path.write_text(
        json.dumps(
            {
                "10": {"inputs": {"image": "old.png"}},
                "11": {"inputs": {"video": "old.mp4"}},
                "12": {"inputs": {"text": "old prompt"}},
                "13": {"inputs": {"orientation": "image"}},
            }
        ),
        encoding="utf-8",
    )
    input_dir = tmp_path / "comfy-input"
    input_dir.mkdir()
    monkeypatch.setattr(settings, "workflow_path", workflow_path)
    monkeypatch.setattr(settings, "comfyui_input_dir", input_dir)
    monkeypatch.setattr(settings, "image_node_id", "10")
    monkeypatch.setattr(settings, "video_node_id", "11")
    monkeypatch.setattr(settings, "prompt_node_id", "12")
    monkeypatch.setattr(settings, "image_input_name", "image")
    monkeypatch.setattr(settings, "video_input_name", "video")
    monkeypatch.setattr(settings, "prompt_input_name", "text")
    monkeypatch.setattr(settings, "orientation_node_id", "13")
    monkeypatch.setattr(settings, "orientation_input_name", "orientation")
    monkeypatch.setattr(settings, "comfy_timeout_seconds", 5)
    return input_dir


def test_fake_comfy_receives_configured_inputs_and_downloads_video(tmp_path, monkeypatch, comfy_server):
    input_dir = _configure_workflow(tmp_path, monkeypatch)
    image = tmp_path / "character.png"
    image.write_bytes(b"image bytes")
    video = tmp_path / "reference.mp4"
    video.write_bytes(b"video bytes")
    output_dir = tmp_path / "job"
    output_dir.mkdir()
    progress = []

    output = comfy.generate(
        image,
        video,
        "A character dances.",
        output_dir,
        "video",
        lambda percent, message: progress.append((percent, message)),
        lambda: False,
    )

    assert output.read_bytes() == b"fake-mp4-content"
    body = FakeComfyHandler.prompt_body
    assert body is not None
    workflow = body["prompt"]
    assert workflow["10"]["inputs"]["image"] == image.name.replace("character", "motion-studio-" + output_dir.name)
    assert workflow["11"]["inputs"]["video"] == video.name.replace("reference", "motion-studio-" + output_dir.name)
    assert workflow["12"]["inputs"]["text"] == (
        "A character dances. Match the character's facing/orientation to the motion reference video."
    )
    assert workflow["13"]["inputs"]["orientation"] == "video"
    assert (input_dir / workflow["10"]["inputs"]["image"]).read_bytes() == b"image bytes"
    assert progress[-1][0] == 92


def test_fake_comfy_reports_out_of_memory(tmp_path, monkeypatch, comfy_server):
    _configure_workflow(tmp_path, monkeypatch)
    FakeComfyHandler.mode = "oom"
    image = tmp_path / "character.png"
    video = tmp_path / "reference.mp4"
    image.write_bytes(b"image")
    video.write_bytes(b"video")
    output_dir = tmp_path / "job"
    output_dir.mkdir()

    with pytest.raises(RuntimeError, match="ran out of GPU memory"):
        comfy.generate(image, video, "prompt", output_dir, "image", lambda *_: None, lambda: False)


def test_fake_comfy_timeout_is_plainly_reported(tmp_path, monkeypatch, comfy_server):
    _configure_workflow(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "comfy_timeout_seconds", -1)
    image = tmp_path / "character.png"
    video = tmp_path / "reference.mp4"
    image.write_bytes(b"image")
    video.write_bytes(b"video")
    output_dir = tmp_path / "job"
    output_dir.mkdir()

    with pytest.raises(RuntimeError, match="ComfyUI timed out"):
        comfy.generate(image, video, "prompt", output_dir, "image", lambda *_: None, lambda: False)


def test_lite_limits_cpu_pose_tracking_to_ten_frames_per_second():
    assert lite._sample_stride(30) == 3
    assert lite._sample_stride(24) == 3
    assert lite._sample_stride(10) == 1
    assert lite._sample_stride(8) == 1
