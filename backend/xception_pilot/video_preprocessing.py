"""YuNet alignment adapter for the DeepGuard Xception validation pilot.

The target template and expansion reproduce DeepfakeBench v1.0.1 geometry.
YuNet landmarks DO NOT reproduce dlib's selected landmark positions. This is
an explicitly different preprocessing pipeline requiring its own evaluation.
See VIDEO_PILOT_README.md for sources, assumptions, and limitations.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np

YUNET_SHA256 = "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"
PREPROCESSING_VERSION = "deepguard-yunet-xception-align-v1"
SIZE = 256
TARGET_FRAMES = 16
MIN_FACES = 8
DETECTOR_THRESHOLD = 0.85


def reference_template() -> np.ndarray:
    """Reference destination geometry, including the 1.3 expansion.

    Adapted from SCLBD/DeepfakeBench, v1.0.1, preprocessing/preprocess.py,
    extract_aligned_face_dlib -> img_align_crop (author: Zhiyuan Yan).
    The existing pilot includes the upstream license and attribution.
    """
    points = np.array([[30.2946, 51.6963], [65.5318, 51.5014],
                       [48.0252, 71.7366], [33.5493, 92.3655],
                       [62.7299, 92.2041]], dtype=np.float32)
    points[:, 0] += 8.0
    points *= SIZE / 112.0
    margin = SIZE * (1.3 - 1.0) / 2.0
    points += margin
    points *= SIZE / (SIZE + 2 * margin)
    return points


def similarity_matrix(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    """Least-squares scale/rotation/translation, with no shear or reflection."""
    source = np.asarray(source, dtype=np.float64)
    destination = np.asarray(destination, dtype=np.float64)
    if source.shape != (5, 2) or destination.shape != (5, 2):
        raise ValueError("Five pairs of face landmarks are required.")
    if not np.isfinite(source).all() or not np.isfinite(destination).all():
        raise ValueError("Non-finite face landmarks.")
    design = np.zeros((10, 4), dtype=np.float64)
    x, y = source.T
    design[0::2] = np.column_stack((x, -y, np.ones(5), np.zeros(5)))
    design[1::2] = np.column_stack((y, x, np.zeros(5), np.ones(5)))
    solution, _, rank, _ = np.linalg.lstsq(design, destination.ravel(), rcond=None)
    a, b, tx, ty = solution
    if rank < 4 or not np.isfinite(solution).all() or math.hypot(a, b) < 1e-6:
        raise ValueError("Degenerate alignment transform.")
    return np.array([[a, -b, tx], [b, a, ty]], dtype=np.float64)


def rgb_normalized_chw(crop_bgr: np.ndarray) -> np.ndarray:
    """uint8 BGR 256x256 -> float32 RGB CHW, (pixel/255 - 0.5)/0.5."""
    if crop_bgr.shape != (SIZE, SIZE, 3) or crop_bgr.dtype != np.uint8:
        raise ValueError("Expected a uint8 BGR 256x256 aligned face.")
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - 0.5) / 0.5
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


def make_detector(model_path: Path):
    if not model_path.is_file():
        raise ValueError(f"YuNet file missing: {model_path}")
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != YUNET_SHA256:
        raise ValueError("YuNet checksum differs from the verified 2026may model. "
                         "Report this error; do not rename another model to bypass it.")
    return cv2.FaceDetectorYN.create(str(model_path), "", (320, 320),
                                     DETECTOR_THRESHOLD, 0.3, 5000)


def thumbnail(frame: np.ndarray) -> np.ndarray:
    out = np.full((160, 256, 3), 24, dtype=np.uint8)
    h, w = frame.shape[:2]
    scale = min(256 / w, 160 / h)
    resized = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    rh, rw = resized.shape[:2]
    y, x = (160 - rh) // 2, (256 - rw) // 2
    out[y:y + rh, x:x + rw] = resized
    return out


def align_single_face(frame: np.ndarray, detector) -> tuple[np.ndarray | None, dict, np.ndarray]:
    """Detect on a downscaled frame, align from the original resolution."""
    h, w = frame.shape[:2]
    if min(h, w) < 64:
        return None, {"status": "small_frame"}, thumbnail(frame)
    scale = min(1.0, 640.0 / max(h, w))
    dw, dh = max(1, round(w * scale)), max(1, round(h * scale))
    small = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
    detector.setInputSize((dw, dh))
    _, detections = detector.detect(small)
    count = 0 if detections is None else len(detections)
    record = {"detected_faces": count}
    if count != 1:
        record["status"] = "no_face" if count == 0 else "multiple_faces"
        return None, record, thumbnail(frame)
    det = np.asarray(detections[0], dtype=np.float64)
    if det.shape != (15,) or not np.isfinite(det).all():
        return None, {**record, "status": "invalid_detection"}, thumbnail(frame)
    sx, sy = w / dw, h / dh
    box = det[:4] * np.array([sx, sy, sx, sy])
    # YuNet: person's right eye, left eye, nose, right mouth, left mouth.
    # For a frontal face these are image-left eye/mouth first, matching
    # the reference destination order. Do NOT swap left/right by name.
    landmarks = det[4:14].reshape(5, 2) * np.array([sx, sy])
    record.update({"detector_score": float(det[14]), "box_xywh": box.tolist(),
                   "landmarks_xy": landmarks.tolist()})
    marked = frame.copy()
    bx, by, bw, bh = np.rint(box).astype(int)
    cv2.rectangle(marked, (bx, by), (bx + bw, by + bh), (0, 220, 0), 2)
    for index, point in enumerate(landmarks):
        xy = tuple(np.rint(point).astype(int))
        cv2.circle(marked, xy, 3, (0, 220, 255), -1)
        cv2.putText(marked, str(index), xy, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    thumb = thumbnail(marked)
    if min(box[2:]) < 32 or np.linalg.norm(landmarks[1] - landmarks[0]) < 10:
        return None, {**record, "status": "small_face"}, thumb
    if ((landmarks[:, 0] < 0).any() or (landmarks[:, 0] >= w).any()
            or (landmarks[:, 1] < 0).any() or (landmarks[:, 1] >= h).any()):
        return None, {**record, "status": "landmarks_outside_frame"}, thumb
    try:
        target = reference_template()
        matrix = similarity_matrix(landmarks, target)
    except ValueError:
        return None, {**record, "status": "invalid_alignment"}, thumb
    fitted = landmarks @ matrix[:, :2].T + matrix[:, 2]
    rms = float(np.sqrt(np.mean(np.sum((fitted - target) ** 2, axis=1))))
    record["alignment_rms_pixels"] = rms
    if rms > 20.0:
        return None, {**record, "status": "poor_landmark_fit"}, thumb
    # Match the reference warp's linear interpolation and zero border.
    crop = cv2.warpAffine(frame, matrix, (SIZE, SIZE), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.warpAffine(np.ones((h, w), dtype=np.uint8), matrix, (SIZE, SIZE),
                           flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    padding = 1.0 - float(valid.mean())
    record["padding_fraction"] = padding
    if padding > 0.25:
        return None, {**record, "status": "excessive_padding"}, thumb
    return crop, {**record, "status": "accepted"}, thumb


def preprocess_video(video_path: Path, detector) -> tuple[list, list, list, dict]:
    """Decode only requested frames. This is not a full corruption scan."""
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise ValueError(f"Video file missing or empty: {video_path}")
    if video_path.suffix.lower() != ".mp4":
        raise ValueError("This pilot accepts MP4 files.")
    if video_path.stat().st_size > 200 * 1024 * 1024:
        raise ValueError("Pilot limit: 200 MiB per video.")
    cap = cv2.VideoCapture(str(video_path))
    crops, records, thumbs = [], [], []
    try:
        if not cap.isOpened():
            raise ValueError("The video could not be opened as a video.")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        raw_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = (float(cap.get(prop)) for prop in
                         (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        if not all(math.isfinite(x) and x > 0 for x in [fps, raw_count, width, height]):
            raise ValueError("Missing or invalid video metadata; no result produced.")
        count = int(raw_count)
        if count < 1 or count > 50_000 or raw_count / fps > 120:
            raise ValueError("Pilot limit: 120 seconds / 50,000 reported frames.")
        if width * height > 3840 * 2160:
            raise ValueError("Pilot limit: 3840x2160 total pixels per frame.")
        indices = np.linspace(0, count - 1, min(TARGET_FRAMES, count), dtype=int).tolist()
        for index in indices:
            record = {"frame_index": index}
            crop = None
            thumb = np.full((160, 256, 3), 24, dtype=np.uint8)
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, index):
                record["status"] = "seek_failed"
            else:
                ok, frame = cap.read()
                position = float(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if not ok or frame is None or frame.size == 0:
                    record["status"] = "decode_failed"
                elif not math.isfinite(position) or abs(position - (index + 1)) > 0.5:
                    record["status"] = "seek_mismatch"
                elif frame.shape[0] * frame.shape[1] > 3840 * 2160:
                    record["status"] = "frame_too_large"
                else:
                    crop, result, thumb = align_single_face(frame, detector)
                    record.update(result)
            if crop is not None:
                record["crop_index"] = len(crops)
                crops.append(crop)
            records.append(record)
            thumbs.append(thumb)
            print(f"  Frame {index}: {record['status']}", flush=True)
        metadata = {
            "reported_width": int(width), "reported_height": int(height), "fps": fps,
            "reported_frame_count": count, "estimated_duration_seconds": raw_count / fps,
            "sampling_method": "uniform_reported_frame_indices", "target_frames": TARGET_FRAMES,
            "requested_frame_indices": indices, "full_decode_checked": False,
            "counts": dict(Counter(row["status"] for row in records)),
        }
        return crops, records, thumbs, metadata
    finally:
        cap.release()


def save_preview(path: Path, crops: list, records: list, thumbs: list) -> None:
    """Top of each tile: sampled frame; bottom: actual aligned model input."""
    columns, tile_h, tile_w = 4, 480, 256
    sheet = np.full((math.ceil(len(records) / columns) * tile_h, columns * tile_w, 3),
                    24, dtype=np.uint8)
    for i, row in enumerate(records):
        y, x = (i // columns) * tile_h, (i % columns) * tile_w
        tile = sheet[y:y + tile_h, x:x + tile_w]
        cv2.putText(tile, f"Frame {row['frame_index']}: {row['status']}", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1)
        tile[24:184] = thumbs[i]
        if "crop_index" in row:
            tile[188:444] = crops[row["crop_index"]]
            score = row.get("fake_score")
            label = "Aligned face: 256 x 256" if score is None else f"Fake score: {score:.4f}"
            cv2.putText(tile, label, (5, 464), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1)
    ok, encoded = cv2.imencode(".png", sheet)
    if not ok:
        raise ValueError("Could not encode preview image.")
    encoded.tofile(str(path))
