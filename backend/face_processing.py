"""Detect and crop one face per sampled frame using OpenCV Zoo's YuNet.

This module locates faces. It does not classify videos as real or fake.
The caller owns the temporary directory and removes its contents.
"""

import logging
import math
from pathlib import Path

import cv2


LOGGER = logging.getLogger(__name__)
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "face_detection_yunet_2026may.onnx"
)
SCORE_THRESHOLD = 0.85
DETECTION_MAX_SIDE = 640
CROP_SIZE = 224
CROP_MARGIN = 0.20  # Increase the square's side by 20% in total.


class FaceProcessingError(RuntimeError):
    pass


class FaceModelError(FaceProcessingError):
    pass


def _create_detector():
    if not MODEL_PATH.is_file():
        raise FaceModelError(
            "YuNet model not found. Check the backend/models folder."
        )

    try:
        # Each request gets its own detector; no mutable detector is shared.
        return cv2.FaceDetectorYN.create(
            str(MODEL_PATH),
            "",
            (320, 320),
            SCORE_THRESHOLD,
            0.3,
            5000,
            cv2.dnn.DNN_BACKEND_OPENCV,
            cv2.dnn.DNN_TARGET_CPU,
        )
    except cv2.error as error:
        LOGGER.exception("YuNet could not be loaded")
        raise FaceModelError(
            "YuNet could not be loaded. Check the backend terminal."
        ) from error


def _square_crop(frame, box):
    """Crop original pixels, pad at image edges, then resize to 224 x 224."""
    height, width = frame.shape[:2]
    x, y, box_width, box_height = map(float, box)

    if not all(math.isfinite(v) for v in (x, y, box_width, box_height)):
        return None
    if box_width <= 0 or box_height <= 0:
        return None

    left = max(0, min(width, math.floor(x)))
    top = max(0, min(height, math.floor(y)))
    right = max(0, min(width, math.ceil(x + box_width)))
    bottom = max(0, min(height, math.ceil(y + box_height)))

    if right <= left or bottom <= top:
        return None

    side = math.ceil(max(right - left, bottom - top) * (1 + CROP_MARGIN))
    x1 = math.floor((left + right - side) / 2)
    y1 = math.floor((top + bottom - side) / 2)
    x2, y2 = x1 + side, y1 + side

    crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    crop = cv2.copyMakeBorder(
        crop,
        max(0, -y1),
        max(0, y2 - height),
        max(0, -x1),
        max(0, x2 - width),
        cv2.BORDER_REPLICATE,
    )

    interpolation = cv2.INTER_AREA if side >= CROP_SIZE else cv2.INTER_LINEAR
    return cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=interpolation)


def detect_and_crop_faces(
    frames: list[tuple[int, Path]],
    output_dir: Path,
) -> tuple[list[tuple[int, Path]], dict]:
    """Return crop paths and counts; skip missing or ambiguous detections.

    Paths are valid only while the caller's temporary directory exists.
    OpenCV arrays use BGR; model-specific RGB conversion/normalization is
    deliberately deferred until a deepfake classifier is integrated.
    """
    detector = _create_detector()
    output_dir.mkdir(parents=True, exist_ok=False)
    crops = []
    skipped_no_face = 0
    skipped_multiple_faces = 0
    skipped_invalid_box = 0
    skipped_small_frame = 0

    try:
        for frame_index, frame_path in frames:
            frame = cv2.imread(str(frame_path))
            if frame is None or frame.size == 0:
                raise FaceProcessingError(
                    "A temporary frame could not be read on the server."
                )

            height, width = frame.shape[:2]
            if min(height, width) < 32:
                skipped_small_frame += 1
                continue

            scale = min(1.0, DETECTION_MAX_SIDE / max(width, height))
            detect_width = max(1, round(width * scale))
            detect_height = max(1, round(height * scale))
            detection_image = cv2.resize(
                frame,
                (detect_width, detect_height),
                interpolation=cv2.INTER_AREA,
            )

            detector.setInputSize((detect_width, detect_height))
            _, faces = detector.detect(detection_image)
            face_count = 0 if faces is None else len(faces)

            if face_count == 0:
                skipped_no_face += 1
                continue
            if face_count > 1:
                skipped_multiple_faces += 1
                continue

            # Map the detected box back to the original-resolution image.
            x, y, box_width, box_height = map(float, faces[0, :4])
            box = (
                x * width / detect_width,
                y * height / detect_height,
                box_width * width / detect_width,
                box_height * height / detect_height,
            )
            crop = _square_crop(frame, box)
            if crop is None:
                skipped_invalid_box += 1
                continue

            crop_path = output_dir / f"face_{frame_index:06d}.png"
            if not cv2.imwrite(str(crop_path), crop):
                raise OSError("A temporary face crop could not be written.")
            crops.append((frame_index, crop_path))

    except cv2.error as error:
        LOGGER.exception("Face preprocessing failed")
        raise FaceProcessingError(
            "Face preprocessing failed. Check the backend terminal."
        ) from error

    return crops, {
        "status": "completed" if crops else "no_usable_faces",
        "detector": "YuNet",
        "model": MODEL_PATH.name,
        "score_threshold": SCORE_THRESHOLD,
        "input_frames": len(frames),
        "crops_created": len(crops),
        "skipped_no_face": skipped_no_face,
        "skipped_multiple_faces": skipped_multiple_faces,
        "skipped_invalid_box": skipped_invalid_box,
        "skipped_small_frame": skipped_small_frame,
        "cropped_frame_indices": [index for index, _ in crops],
        "crop_size": [CROP_SIZE, CROP_SIZE],
        "crop_margin_ratio": CROP_MARGIN,
        "edge_padding": "replicate",
        "selection_policy": "single_face_only",
        "face_tracking_performed": False,
        "crops_retained": False,
    }
