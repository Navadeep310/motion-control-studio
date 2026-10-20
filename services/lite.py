from __future__ import annotations

import math
from pathlib import Path
from typing import Callable


Progress = Callable[[int, str], None]


def generate(
    image_path: Path,
    video_path: Path,
    destination: Path,
    progress: Progress,
    cancelled: Callable[[], bool],
) -> Path:
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Lite mode dependencies are missing. Install them with "
            "`pip install -r requirements.txt`, then restart the app."
        ) from exc

    source = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if source is None:
        raise RuntimeError("OpenCV could not read the character image.")
    if source.ndim == 2:
        source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGRA)
    elif source.shape[2] == 3:
        source = cv2.cvtColor(source, cv2.COLOR_BGR2BGRA)
    source_h, source_w = source.shape[:2]

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open the reference video.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    stride = _sample_stride(fps)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("The reference video has invalid dimensions.")
    output_w = min(1280, max(320, width))
    output_h = max(1, round(height * output_w / width))
    temporary = destination.with_name("lite_no_audio.mp4")
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / stride,
        (output_w, output_h),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("OpenCV could not create the preview video. Check available codecs/disk space.")

    pose_api = mp.solutions.pose
    source_landmarks = _source_pose(source_w, source_h)
    body_mask = _person_mask(source, cv2, np)
    try:
        with pose_api.Pose(
            static_image_mode=False,
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            frame_index = 0
            rendered_frames = 0
            output_frame_count = max(1, math.ceil(frame_count / stride))
            while True:
                if cancelled():
                    raise InterruptedError("Generation was canceled.")
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % stride:
                    frame_index += 1
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                canvas = np.zeros((output_h, output_w, 4), dtype=np.uint8)
                if result.pose_landmarks:
                    target = _target_pose(result.pose_landmarks.landmark, output_w, output_h)
                    _render_pose(canvas, source, body_mask, source_landmarks, target, cv2, np)
                else:
                    _render_still(canvas, source, body_mask, cv2, np)
                bgr = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
                writer.write(bgr)
                rendered_frames += 1
                frame_index += 1
                if rendered_frames % max(1, output_frame_count // 20) == 0:
                    progress(
                        min(86, 15 + int(70 * rendered_frames / output_frame_count)),
                        f"Tracking pose and rendering preview "
                        f"({rendered_frames}/{output_frame_count} frames)",
                    )
    except InterruptedError:
        raise
    finally:
        capture.release()
        writer.release()

    if rendered_frames == 0:
        raise RuntimeError("The reference video contained no decodable frames.")
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError("OpenCV did not produce a preview video.")
    return temporary


def _sample_stride(fps: float, target_fps: float = 10.0) -> int:
    return max(1, math.ceil(fps / target_fps))


def _person_mask(source, cv2, np):
    height, width = source.shape[:2]
    if source.shape[2] == 4 and (source[:, :, 3] < 250).any():
        return source[:, :, 3].copy()
    bgr = source[:, :, :3]
    scale = min(1.0, 768 / max(height, width))
    if scale < 1:
        segmentation = cv2.resize(
            bgr,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        segmentation = bgr
    mask_height, mask_width = segmentation.shape[:2]
    gc_mask = np.zeros((mask_height, mask_width), dtype="uint8")
    bg_model = np.zeros((1, 65), dtype="float64")
    fg_model = np.zeros((1, 65), dtype="float64")
    rectangle = (
        max(1, int(mask_width * 0.04)),
        max(1, int(mask_height * 0.02)),
        max(2, int(mask_width * 0.92)),
        max(2, int(mask_height * 0.96)),
    )
    try:
        cv2.grabCut(
            segmentation, gc_mask, rectangle, bg_model, fg_model, 3, cv2.GC_INIT_WITH_RECT
        )
        mask = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
        ).astype("uint8")
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        if scale < 1:
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        return mask
    except cv2.error as exc:
        raise RuntimeError(
            "Lite mode could not separate the character from the image background. "
            "Try a PNG with a transparent background or a plain contrasting background."
        ) from exc


def _source_pose(width: int, height: int) -> dict[str, tuple[float, float]]:
    return {
        "nose": (width * 0.50, height * 0.13),
        "left_shoulder": (width * 0.36, height * 0.30),
        "right_shoulder": (width * 0.64, height * 0.30),
        "left_elbow": (width * 0.25, height * 0.46),
        "right_elbow": (width * 0.75, height * 0.46),
        "left_wrist": (width * 0.18, height * 0.62),
        "right_wrist": (width * 0.82, height * 0.62),
        "left_hip": (width * 0.42, height * 0.57),
        "right_hip": (width * 0.58, height * 0.57),
        "left_knee": (width * 0.38, height * 0.77),
        "right_knee": (width * 0.62, height * 0.77),
        "left_ankle": (width * 0.36, height * 0.96),
        "right_ankle": (width * 0.64, height * 0.96),
    }


def _target_pose(landmarks, width: int, height: int) -> dict[str, tuple[float, float]]:
    indices = {
        "nose": 0,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
    }
    body_width = max(landmarks[11].x, landmarks[12].x) - min(landmarks[11].x, landmarks[12].x)
    scale = min(1.5, max(0.5, 0.55 / max(body_width, 0.05)))
    center_x = (landmarks[23].x + landmarks[24].x) / 2
    center_y = (landmarks[11].y + landmarks[12].y + landmarks[23].y + landmarks[24].y) / 4
    return {
        name: (
            width / 2 + (landmarks[index].x - center_x) * width * scale,
            height * 0.52 + (landmarks[index].y - center_y) * height * scale,
        )
        for name, index in indices.items()
    }


def _person_masked_source(source, mask, cv2):
    rgba = source.copy()
    rgba[:, :, 3] = (rgba[:, :, 3].astype("float32") * mask / 255).astype("uint8")
    return rgba


def _render_pose(canvas, source, mask, src, dst, cv2, np):
    rgba_source = _person_masked_source(source, mask, cv2)
    torso_src = np.float32(
        [src["left_hip"], src["right_hip"], src["right_shoulder"], src["left_shoulder"]]
    )
    torso_dst = np.float32(
        [dst["left_hip"], dst["right_hip"], dst["right_shoulder"], dst["left_shoulder"]]
    )
    _warp_polygon(canvas, rgba_source, torso_src, torso_dst, cv2, np)
    head_src = np.float32(
        [
            src["nose"],
            (src["nose"][0] - source.shape[1] * 0.10, src["nose"][1]),
            (src["nose"][0], src["nose"][1] - source.shape[1] * 0.10),
        ]
    )
    head_dst = np.float32(
        [
            dst["nose"],
            (dst["nose"][0] - source.shape[1] * 0.10, dst["nose"][1]),
            (dst["nose"][0], dst["nose"][1] - source.shape[1] * 0.10),
        ]
    )
    _warp_polygon(canvas, rgba_source, head_src, head_dst, cv2, np)
    for first, middle, last in (
        ("left_shoulder", "left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow", "right_wrist"),
        ("left_hip", "left_knee", "left_ankle"),
        ("right_hip", "right_knee", "right_ankle"),
    ):
        for a, b in ((first, middle), (middle, last)):
            _warp_segment(canvas, rgba_source, src[a], src[b], dst[a], dst[b], cv2, np)


def _warp_polygon(canvas, source, src_points, dst_points, cv2, np):
    matrix = cv2.getAffineTransform(src_points[:3], dst_points[:3])
    warped = cv2.warpAffine(source, matrix, (canvas.shape[1], canvas.shape[0]))
    _blend(canvas, warped, cv2, np)


def _warp_segment(canvas, source, src_a, src_b, dst_a, dst_b, cv2, np):
    source_vector = np.float32([src_b[0] - src_a[0], src_b[1] - src_a[1]])
    target_vector = np.float32([dst_b[0] - dst_a[0], dst_b[1] - dst_a[1]])
    source_length = max(1.0, float(np.linalg.norm(source_vector)))
    target_length = max(1.0, float(np.linalg.norm(target_vector)))
    scale = target_length / source_length
    angle = math.atan2(target_vector[1], target_vector[0]) - math.atan2(
        source_vector[1], source_vector[0]
    )
    cosine, sine = math.cos(angle) * scale, math.sin(angle) * scale
    matrix = np.float32(
        [
            [cosine, -sine, dst_a[0] - cosine * src_a[0] + sine * src_a[1]],
            [sine, cosine, dst_a[1] - sine * src_a[0] - cosine * src_a[1]],
        ]
    )
    warped = cv2.warpAffine(source, matrix, (canvas.shape[1], canvas.shape[0]))
    _blend(canvas, warped, cv2, np)


def _blend(canvas, layer, cv2, np):
    alpha = layer[:, :, 3:4].astype("float32") / 255
    canvas[:, :, :3] = (
        layer[:, :, :3].astype("float32") * alpha
        + canvas[:, :, :3].astype("float32") * (1 - alpha)
    ).astype("uint8")
    canvas[:, :, 3:4] = np.maximum(canvas[:, :, 3:4], layer[:, :, 3:4])


def _render_still(canvas, source, mask, cv2, np):
    source_h, source_w = source.shape[:2]
    scale = min(canvas.shape[1] * 0.7 / source_w, canvas.shape[0] * 0.9 / source_h)
    width, height = round(source_w * scale), round(source_h * scale)
    resized = cv2.resize(source, (width, height))
    resized[:, :, 3] = cv2.resize(mask, (width, height))
    x, y = (canvas.shape[1] - width) // 2, (canvas.shape[0] - height) // 2
    _blend(canvas[y : y + height, x : x + width], resized, cv2, np)
