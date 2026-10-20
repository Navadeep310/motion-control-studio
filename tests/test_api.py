from __future__ import annotations

from fastapi.testclient import TestClient

import main
from config import settings
from tests.fakes import FakeOllamaHandler, FakeServer, sample_image_bytes


def test_completed_result_has_inline_playback_and_attachment_download(tmp_path):
    result = tmp_path / "result.mp4"
    result.write_bytes(b"test mp4")
    job_id = "playback-test"
    main.jobs[job_id] = main.Job(
        job_id,
        "prepared-test",
        "lite",
        "prompt",
        state="completed",
        result=result,
    )
    try:
        with TestClient(main.app) as client:
            status = client.get(f"/api/jobs/{job_id}")
            video = client.get(f"/api/jobs/{job_id}/video")
            download = client.get(f"/api/jobs/{job_id}/download")
        assert status.json()["video_url"] == f"/api/jobs/{job_id}/video"
        assert video.headers["content-disposition"].startswith("inline;")
        assert video.content == b"test mp4"
        assert download.headers["content-disposition"].startswith("attachment;")
    finally:
        main.jobs.pop(job_id, None)


def test_prepare_uses_fake_ollama_and_returns_editable_prompt(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(main.media, "image_is_valid", lambda path: True)
    monkeypatch.setattr(main.media, "video_duration", lambda path: 4.5)
    monkeypatch.setattr(main.media, "sample_video_frames", lambda path, directory, count: [
        _write_frame(directory, index) for index in range(count)
    ])
    FakeOllamaHandler.result = {
        "person_visible": True,
        "character_description": "A character in a red jacket.",
        "motion_description": "The subject turns and waves.",
        "generation_prompt": "A red-jacketed character turns and waves, smooth camera.",
    }
    with FakeServer(FakeOllamaHandler) as url:
        monkeypatch.setattr(settings, "ollama_url", url)
        with TestClient(main.app) as client:
            response = client.post(
                "/api/prepare",
                files={
                    "character": (
                        "character.png",
                        sample_image_bytes(".png"),
                        "image/png",
                    ),
                    "motion": ("motion.mp4", b"video", "video/mp4"),
                },
                data={"orientation": "image", "keep_audio": "true"},
            )
    assert response.status_code == 200
    body = response.json()
    assert body["prepared_id"]
    assert body["person_visible"] is True
    assert body["generation_prompt"] == "A red-jacketed character turns and waves, smooth camera."


def _write_frame(directory, index):
    directory.mkdir(parents=True, exist_ok=True)
    frame = directory / f"frame-{index}.jpg"
    frame.write_bytes(sample_image_bytes(".jpg"))
    return frame


def test_prepare_rejects_unsupported_file_types():
    with TestClient(main.app) as client:
        response = client.post(
            "/api/prepare",
            files={
                "character": ("character.gif", b"image", "image/gif"),
                "motion": ("motion.webm", b"video", "video/webm"),
            },
        )
    assert response.status_code == 415
    assert "PNG or JPG" in response.json()["detail"]


def test_prepare_rejects_video_over_ten_seconds(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(main.media, "image_is_valid", lambda path: True)
    monkeypatch.setattr(main.media, "video_duration", lambda path: 10.1)
    with TestClient(main.app) as client:
        response = client.post(
            "/api/prepare",
            files={
                "character": ("character.png", sample_image_bytes(".png"), "image/png"),
                "motion": ("motion.mp4", b"video", "video/mp4"),
            },
        )
    assert response.status_code == 413
    assert "maximum is 10 seconds" in response.json()["detail"]
