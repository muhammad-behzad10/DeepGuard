"""Create a local contact sheet with the project's actual sampled face crops.

Place this file beside main.py in the existing DeepGuard backend folder.
Run: python preview_faces.py --video "C:\\path\\to\\001.mp4"

The final preview PNG is intentionally retained in backend/debug_previews.
Intermediate frames and crops are temporary. The source video is read only.
"""

import argparse
import math
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import cv2
import numpy as np

from face_processing import CROP_SIZE, FaceProcessingError, detect_and_crop_faces
from frame_extraction import extract_frames
from video_validation import VideoValidationError, validate_video


def make_contact_sheet(frames, crops, face_info):
    """Show each sampled frame's crop, or a labeled empty tile if skipped."""
    columns = 4
    rows = math.ceil(len(frames) / columns)
    margin, header, footer = 18, 110, 75
    tile_width, tile_height = CROP_SIZE + 24, CROP_SIZE + 52
    width = columns * tile_width + 2 * margin
    height = header + rows * tile_height + footer
    sheet = np.full((height, width, 3), (246, 247, 249), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    ink = (54, 45, 36)

    cv2.putText(
        sheet, "DeepGuard | Face crop preview", (margin, 38),
        font, 0.85, ink, 2, cv2.LINE_AA,
    )
    cv2.putText(
        sheet,
        f"{len(frames)} sampled frames | {len(crops)} crops | "
        f"{CROP_SIZE} x {CROP_SIZE} pixels",
        (margin, 72), font, 0.6, ink, 1, cv2.LINE_AA,
    )

    crop_paths = dict(crops)
    for slot, (frame_index, _) in enumerate(frames):
        column, row = slot % columns, slot // columns
        x = margin + column * tile_width
        y = header + row * tile_height
        sheet[y:y + tile_height - 10, x:x + tile_width - 8] = (255, 255, 255)
        image_x, image_y = x + 8, y + 8
        path = crop_paths.get(frame_index)

        if path is not None:
            crop = cv2.imread(str(path))
            if crop is None or crop.shape != (CROP_SIZE, CROP_SIZE, 3):
                raise ValueError("A temporary face crop has an unexpected format.")
            sheet[
                image_y:image_y + CROP_SIZE,
                image_x:image_x + CROP_SIZE,
            ] = crop
        else:
            sheet[
                image_y:image_y + CROP_SIZE,
                image_x:image_x + CROP_SIZE,
            ] = (233, 236, 239)
            cv2.putText(
                sheet, "NO CROP", (image_x + 50, image_y + CROP_SIZE // 2),
                font, 0.65, ink, 1, cv2.LINE_AA,
            )

        cv2.putText(
            sheet, f"Frame {frame_index}", (image_x, image_y + CROP_SIZE + 25),
            font, 0.55, ink, 1, cv2.LINE_AA,
        )

    skipped = (
        f"Skipped: no face {face_info['skipped_no_face']} | "
        f"multiple faces {face_info['skipped_multiple_faces']} | "
        f"invalid box {face_info['skipped_invalid_box']} | "
        f"small frame {face_info['skipped_small_frame']}"
    )
    cv2.putText(
        sheet, skipped, (margin, height - 45),
        font, 0.5, ink, 1, cv2.LINE_AA,
    )
    cv2.putText(
        sheet, "Preprocessing preview - no Real/Fake prediction",
        (margin, height - 18), font, 0.5, ink, 1, cv2.LINE_AA,
    )
    return sheet


def create_preview(video_path: Path, output_dir: Path):
    if not video_path.is_file():
        raise ValueError("Video file not found. Check the full path.")
    if video_path.suffix.lower() != ".mp4":
        raise ValueError("Please select an MP4 video.")
    if video_path.stat().st_size > 30_000_000:
        raise ValueError("Please select a video within the 30 MB limit.")

    video_info = validate_video(video_path)
    with TemporaryDirectory(prefix="deepguard_preview_") as work_dir:
        frames = extract_frames(
            video_path=video_path,
            output_dir=Path(work_dir) / "frames",
            frame_count=video_info["reported_frame_count"],
        )
        crops, face_info = detect_and_crop_faces(
            frames=frames,
            output_dir=Path(work_dir) / "faces",
        )
        sheet = make_contact_sheet(frames, crops, face_info)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"face_preview_{timestamp}_{uuid4().hex[:6]}.png"
        if not cv2.imwrite(str(output_path), sheet):
            raise OSError("The preview PNG could not be saved.")

    return output_path.resolve(), face_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "debug_previews",
    )
    args = parser.parse_args()
    try:
        output_path, info = create_preview(args.video, args.output_dir)
    except (OSError, ValueError, VideoValidationError, FaceProcessingError, cv2.error) as error:
        parser.exit(1, f"Preview failed: {error}\n")

    print(f"Sampled frames: {info['input_frames']}")
    print(f"Face crops: {info['crops_created']}")
    print("Preview saved:")
    print(output_path)


if __name__ == "__main__":
    main()
