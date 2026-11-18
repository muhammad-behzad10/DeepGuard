from pathlib import Path

import cv2

from video_validation import VideoValidationError


TARGET_FRAMES = 16


def extract_frames(
    video_path: Path,
    output_dir: Path,
    frame_count: int,
    target_frames: int = TARGET_FRAMES,
) -> list[tuple[int, Path]]:
    """Write sampled frames to PNGs in a temporary directory."""
    if frame_count < 1 or target_frames < 1:
        raise ValueError("Frame counts must be positive.")

    sample_count = min(target_frames, frame_count)

    if sample_count == 1:
        positions = [0]
    else:
        positions = [
            i * (frame_count - 1) // (sample_count - 1)
            for i in range(sample_count)
        ]

    output_dir.mkdir(parents=True, exist_ok=False)
    capture = cv2.VideoCapture()
    extracted = []
    expected_shape = None

    try:
        if not capture.open(str(video_path)):
            raise VideoValidationError(
                "The video could not be opened for frame extraction."
            )

        for position in positions:
            if position > 0:
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, position):
                    raise VideoValidationError(
                        "A requested video frame could not be reached."
                    )

            success, frame = capture.read()

            if not success or frame is None or frame.size == 0:
                raise VideoValidationError(
                    "A requested video frame could not be decoded."
                )

            if expected_shape is None:
                expected_shape = frame.shape
            elif frame.shape != expected_shape:
                raise VideoValidationError(
                    "The video resolution changes during playback."
                )

            image_path = output_dir / f"frame_{position:06d}.png"

            try:
                written = cv2.imwrite(str(image_path), frame)
            except cv2.error as error:
                raise OSError(
                    "A temporary frame could not be written."
                ) from error

            if not written:
                raise OSError(
                    "A temporary frame could not be written."
                )

            extracted.append((position, image_path))

        return extracted

    except cv2.error as error:
        raise VideoValidationError(
            "The video could not be decoded for frame extraction."
        ) from error

    finally:
        capture.release()