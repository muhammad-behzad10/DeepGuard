import math
from pathlib import Path

import cv2


MAX_DURATION_SECONDS = 30.0


class VideoValidationError(ValueError):
    pass


def validate_video(path: Path) -> dict:
    capture = cv2.VideoCapture()

    try:
        if not capture.open(str(path)):
            raise VideoValidationError(
                "The video could not be opened. Try another MP4."
            )

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        reported_frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if (
            not math.isfinite(fps)
            or fps <= 0
            or not math.isfinite(reported_frames)
            or reported_frames < 1
        ):
            raise VideoValidationError(
                "The video has missing or invalid timing information."
            )

        frame_count = int(reported_frames)
        duration_estimate = frame_count / fps

        if not math.isfinite(duration_estimate) or duration_estimate <= 0:
            raise VideoValidationError(
                "The video duration could not be estimated."
            )

        if duration_estimate > MAX_DURATION_SECONDS:
            raise VideoValidationError(
                "The estimated video duration exceeds 30 seconds."
            )

        positions = sorted({0, frame_count // 2, frame_count - 1})
        width = height = 0

        for position in positions:
            if position > 0:
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, position):
                    raise VideoValidationError(
                        "The video could not be read at a sampled position."
                    )

            success, frame = capture.read()

            if not success or frame is None or frame.size == 0:
                raise VideoValidationError(
                    "A sampled video frame could not be decoded. "
                    "Try another MP4."
                )

            current_height, current_width = frame.shape[:2]

            if width == 0:
                width, height = int(current_width), int(current_height)
            elif (current_width, current_height) != (width, height):
                raise VideoValidationError(
                    "The video resolution changes during playback. "
                    "Please choose another MP4."
                )

        return {
            "validation": "basic_passed",
            "width": width,
            "height": height,
            "fps": round(fps, 3),
            "reported_frame_count": frame_count,
            "estimated_duration_seconds": round(duration_estimate, 3),
            "frames_checked": len(positions),
            "full_decode_checked": False,
        }

    except cv2.error as error:
        raise VideoValidationError(
            "The video could not be decoded. Try another MP4."
        ) from error

    finally:
        capture.release()