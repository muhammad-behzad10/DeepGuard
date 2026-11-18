"""Standalone CPU video pilot. Does not change the FastAPI service or Flutter."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import uuid

import cv2
import numpy as np
import torch

if __package__:
    from .xception_loader import load_xception, sha256_file, LABELS
    from .video_preprocessing import (make_detector, preprocess_video, rgb_normalized_chw,
                                      save_preview, PREPROCESSING_VERSION, MIN_FACES,
                                      YUNET_SHA256, DETECTOR_THRESHOLD)
else:
    from xception_loader import load_xception, sha256_file, LABELS
    from video_preprocessing import (make_detector, preprocess_video, rgb_normalized_chw,
                                     save_preview, PREPROCESSING_VERSION, MIN_FACES,
                                     YUNET_SHA256, DETECTOR_THRESHOLD)

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent


def aggregate_prediction(scores: list[float], threshold: float = 0.5) -> dict | None:
    """Initial baseline; scores are not calibrated certainty or accuracy."""
    if len(scores) < MIN_FACES:
        return None
    values = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Invalid per-frame model scores.")
    mean_fake = float(values.mean())
    class_id = int(mean_fake >= threshold)
    return {"class_id": class_id, "label": LABELS[class_id], "fake_score": mean_fake,
            "real_score": 1.0 - mean_fake, "decision_threshold": threshold,
            "threshold_status": "initial_baseline_not_tuned",
            "aggregation": "mean_frame_softmax_fake_score", "frames_scored": len(scores),
            "calibrated": False}


def run(args) -> int:
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise ValueError(f"Video file not found: {video}")
    print("DeepGuard Xception video pilot", flush=True)
    print(f"Video: {video.name}", flush=True)
    print(f"Device: cpu | Threads: {args.threads} | Batch size: 1", flush=True)
    print("Loading verified Xception and YuNet...", flush=True)
    model, model_meta = load_xception(args.checkpoint)
    model_meta.pop("preprocessing_implemented_in_this_pilot", None)
    detector = make_detector(args.yunet.resolve())
    loaded = time.perf_counter()
    crops, rows, thumbs, video_meta = preprocess_video(video, detector)
    prepared = time.perf_counter()
    scores = []
    if len(crops) >= MIN_FACES:
        print(f"Running Xception on {len(crops)} aligned faces...", flush=True)
        with torch.inference_mode():
            for row in rows:
                if "crop_index" not in row:
                    continue
                crop = crops[row["crop_index"]]
                tensor = torch.from_numpy(rgb_normalized_chw(crop)).unsqueeze(0)
                logits = model(tensor)
                if tuple(logits.shape) != (1, 2) or not torch.isfinite(logits).all().item():
                    raise ValueError("Model returned invalid logits; no prediction published.")
                probabilities = torch.softmax(logits, dim=1)
                fake_score = float(probabilities[0, 1].item())
                row["fake_score"] = fake_score
                scores.append(fake_score)
    prediction = aggregate_prediction(scores)
    scored = time.perf_counter()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    out = args.output_dir.resolve() / run_id
    out.mkdir(parents=True, exist_ok=False)
    preview_path = out / "face_alignment_preview.png"
    save_preview(preview_path, crops, rows, thumbs)
    report = {
        "schema_version": 1, "status": "scored" if prediction else "insufficient_evidence",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
        "video_path": str(video), "video_sha256": sha256_file(video), "video": video_meta,
        "model": model_meta,
        "runtime": {"python": sys.version.split()[0], "pytorch": torch.__version__,
                    "opencv": cv2.__version__, "numpy": np.__version__,
                    "device": "cpu", "threads": args.threads, "batch_size": 1},
        "preprocessing": {
            "version": PREPROCESSING_VERSION, "detector": "YuNet",
            "detector_sha256": YUNET_SHA256, "detector_threshold": DETECTOR_THRESHOLD,
            "detection_max_side": 640, "selection_policy": "single_face_only",
            "face_tracking_performed": False, "minimum_usable_faces": MIN_FACES,
            "usable_faces": len(crops), "crop_size": [256, 256],
            "colour_order": "RGB", "normalization": "(pixel/255 - 0.5)/0.5",
            "alignment": "5_landmark_least_squares_similarity",
            "template": "DeepfakeBench_v1.0.1_112_template_scaled_to_256_expansion_1.3",
            "warp_interpolation": "linear", "border": "constant_zero",
            "minimum_face_box_side_pixels": 32, "minimum_eye_distance_pixels": 10,
            "maximum_alignment_rms_pixels": 20, "maximum_padding_fraction": 0.25,
            "quality_limits_status": "engineering_defaults_not_validated",
            "identical_to_reference_preprocessing": False,
            "reference_differences": [
                "YuNet 5 landmarks instead of dlib 81-point selected landmarks",
                "single face only instead of largest face",
                "16 uniformly sampled frames instead of reference test configuration 32",
                "additional quality rejection rules"],
        },
        "frames": rows, "inference_performed": bool(scores), "prediction": prediction,
        "expected_label": args.expected_label, "accuracy_evaluated": False,
        "fastapi_integrated": False,
        "timing_seconds": {"model_load": loaded - started,
                           "video_preprocessing": prepared - loaded,
                           "model_inference": scored - prepared,
                           "total_before_output": scored - started},
        "outputs": {"preview_path": str(preview_path), "preview_retained": True,
                    "source_video_copied": False, "individual_crops_retained": False},
        "limitations": ["Model scores are not calibrated authenticity probabilities.",
                        "This pilot is not a benchmark or an accuracy measurement.",
                        "No identity tracking; scene changes can mix people.",
                        "Only sampled frames are decoded.",
                        "No audio or temporal model; frame scores are averaged."]
    }
    report_path = out / "video_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Usable faces: {len(crops)}/{len(rows)} (minimum: {MIN_FACES})", flush=True)
    if prediction:
        print(f"Predicted label: {prediction['label']}", flush=True)
        print(f"Mean Fake score: {prediction['fake_score']:.6f}", flush=True)
        print("Threshold: 0.50 (initial baseline, not tuned)", flush=True)
        if args.expected_label:
            print(f"Dataset label supplied for comparison: {args.expected_label}", flush=True)
    else:
        print("INSUFFICIENT EVIDENCE: no Real/Fake prediction issued.", flush=True)
    print(f"Preprocessing: {prepared - loaded:.2f}s | Inference: {scored - prepared:.2f}s", flush=True)
    print(f"Preview saved:\n{preview_path}", flush=True)
    print(f"Report saved:\n{report_path}", flush=True)
    print("Accuracy has not been measured. FastAPI has not been changed.", flush=True)
    return 0 if prediction else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=BACKEND / "models" / "xception_best.pth")
    parser.add_argument("--yunet", type=Path,
                        default=BACKEND / "models" / "face_detection_yunet_2026may.onnx")
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--output-dir", type=Path, default=HERE / "video_checks")
    parser.add_argument("--expected-label", choices=["Real", "Fake"],
                        help="Report annotation only; never provided to the model.")
    args = parser.parse_args()
    try:
        return run(args)
    except (ValueError, OSError, RuntimeError, cv2.error) as exc:
        print(f"VIDEO PILOT FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
