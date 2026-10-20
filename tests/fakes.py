from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np


def sample_image_bytes(extension: str = ".jpg") -> bytes:
    image = np.full((32, 48, 3), 200, dtype=np.uint8)
    cv2.circle(image, (24, 16), 8, (40, 80, 200), -1)
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise RuntimeError(f"Could not encode test image as {extension}")
    return encoded.tobytes()


class FakeServer:
    def __init__(self, handler):
        self.handler = handler
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    def __enter__(self) -> str:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self.handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}"

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        assert self.server is not None and self.thread is not None
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)


class FakeOllamaHandler(BaseHTTPRequestHandler):
    result: dict[str, Any] = {
        "person_visible": True,
        "character_description": "A red-jacketed character.",
        "motion_description": "The subject raises both arms.",
        "generation_prompt": "A red-jacketed character raises both arms, steady camera.",
    }
    request_json: dict[str, Any] | None = None

    def do_POST(self) -> None:
        if self.path != "/api/generate":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        type(self).request_json = json.loads(self.rfile.read(length))
        body = json.dumps({"response": json.dumps(type(self).result)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


class FakeComfyHandler(BaseHTTPRequestHandler):
    prompt_body: dict[str, Any] | None = None
    mode = "success"

    def do_POST(self) -> None:
        if self.path == "/interrupt":
            self._json({"message": "Interrupted"})
            return
        if self.path != "/prompt":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        type(self).prompt_body = json.loads(self.rfile.read(length))
        self._json({"prompt_id": "fake-prompt-1"})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/history/fake-prompt-1":
            if type(self).mode == "oom":
                self._json(
                    {
                        "fake-prompt-1": {
                            "status": {
                                "status_str": "error",
                                "completed": False,
                                "messages": [
                                    ["execution_error", {"exception_message": "CUDA out of memory"}]
                                ],
                            },
                            "outputs": {},
                        }
                    }
                )
            elif type(self).mode == "pending":
                self._json({})
            else:
                self._json(
                    {
                        "fake-prompt-1": {
                            "status": {"status_str": "success", "completed": True, "messages": []},
                            "outputs": {
                                "99": {
                                    "videos": [
                                        {"filename": "result.mp4", "subfolder": "", "type": "output"}
                                    ]
                                }
                            },
                        }
                    }
                )
        elif parsed.path == "/view":
            params = parse_qs(parsed.query)
            if params.get("filename") != ["result.mp4"]:
                self.send_error(404)
                return
            body = b"fake-mp4-content"
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def _json(self, value: dict[str, Any]) -> None:
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return
