"""Fixed 40-video FF++ validation pilot; reuse the existing Xception adapter.

20 Real videos and 20 Fake videos (5 per method), all from official val IDs.
No training or test video is selected. No model parameters or threshold change.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import random
import re
import sys
import time

import cv2
import numpy as np
import torch

if __package__:
    from .xception_loader import load_xception, sha256_file, verify_checkpoint, EXPECTED_SHA256
    from .video_preprocessing import (make_detector, preprocess_video, rgb_normalized_chw,
                                      PREPROCESSING_VERSION, MIN_FACES, YUNET_SHA256)
    from .predict_video import aggregate_prediction
else:
    from xception_loader import load_xception, sha256_file, verify_checkpoint, EXPECTED_SHA256
    from video_preprocessing import (make_detector, preprocess_video, rgb_normalized_chw,
                                     PREPROCESSING_VERSION, MIN_FACES, YUNET_SHA256)
    from predict_video import aggregate_prediction

HERE = Path(__file__).resolve().parent
SEED = 20261105
METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
# Fingerprints of canonicalized official pairs, checked 2026-11-05.
# Order/whitespace do not matter; changing an ID or pair does matter.
OFFICIAL_SPLIT_HASHES = {
    "train": "b7bf4f27332fcf85c1e16ae269b8b33da86cc0ed42352aa7c353914c2b718656",
    "val": "e085ec121a284fc92744a8ef93557fe256a73a4d98a2c2a071fb66f1a4d2ae14",
    "test": "c7ed1b3d7b22d5e328f223088a000ffe0270e2c4789b6649f434b63ad11b9d67",
}


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_splits(directory: Path) -> tuple[list, dict]:
    split_pairs, ids, fingerprints = {}, {}, {}
    for name, expected_pairs in [("train", 360), ("val", 70), ("test", 70)]:
        path = directory / f"{name}.json"
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(raw, list) or len(raw) != expected_pairs:
            raise ValueError(f"Unexpected pair count in {path}")
        pairs = []
        for pair in raw:
            if (not isinstance(pair, list) or len(pair) != 2
                    or not all(isinstance(s, str) and re.fullmatch(r"\d{3}", s) for s in pair)
                    or pair[0] == pair[1]):
                raise ValueError(f"Invalid pair in {path}")
            pairs.append(sorted(pair))
        pairs.sort()
        flat = [s for pair in pairs for s in pair]
        if len(set(flat)) != len(flat):
            raise ValueError(f"Repeated original IDs inside {path}")
        fingerprint = canonical_hash(pairs)
        if fingerprint != OFFICIAL_SPLIT_HASHES[name]:
            raise ValueError(f"{name}.json differs from the recorded official pairs. "
                             "Report this error; do not replace the fingerprints.")
        split_pairs[name], ids[name], fingerprints[name] = pairs, set(flat), fingerprint
    if any(ids[a] & ids[b] for a, b in [("train", "val"), ("train", "test"), ("val", "test")]):
        raise ValueError("Official split IDs overlap.")
    return split_pairs["val"], fingerprints


def select_subset(val_pairs: list) -> list[dict]:
    """Choose before predictions; pair-group membership never changes."""
    rng = random.Random(SEED)
    chosen = rng.sample(sorted(sorted(p) for p in val_pairs), 10)
    assignments = METHODS * 5
    rng.shuffle(assignments)
    rows = []
    slot = 0
    for a, b in chosen:
        for first, second in [(a, b), (b, a)]:
            rows.append({"relative_path": f"FaceForensics++/original_sequences/youtube/c23/videos/{first}.mp4",
                         "label": 0, "label_name": "Real", "method": "Original",
                         "source_pair": f"{a}_{b}", "split": "val"})
            method = assignments[slot]
            slot += 1
            rows.append({"relative_path": f"FaceForensics++/manipulated_sequences/{method}/c23/videos/{first}_{second}.mp4",
                         "label": 1, "label_name": "Fake", "method": method,
                         "source_pair": f"{a}_{b}", "split": "val"})
    # Mix Real/Fake/method order while keeping a reproducible plan.
    rng.shuffle(rows)
    for i, row in enumerate(rows, 1):
        row["case_id"] = f"case_{i:03d}"
    return rows


def preflight(data_root: Path, rows: list[dict]) -> list[dict]:
    missing = []
    for row in rows:
        path = data_root / row["relative_path"]
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(str(path))
    if missing:
        raise ValueError("Subset preflight failed: missing/empty files:\n" + "\n".join(missing))
    verified = []
    for row in rows:
        path = data_root / row["relative_path"]
        verified.append({**row, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return verified


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def binary_metrics(labels: list[int], scores: list[float]) -> dict:
    if len(labels) != len(scores) or any(y not in (0, 1) for y in labels):
        raise ValueError("Invalid evaluation labels.")
    if any(not math.isfinite(s) or s < 0 or s > 1 for s in scores):
        raise ValueError("Invalid model scores in evaluation.")
    predictions = [int(s >= 0.5) for s in scores]
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    real_n, fake_n = tn + fp, tp + fn
    specificity = tn / real_n if real_n else None
    recall = tp / fake_n if fake_n else None
    # Pairwise ROC-AUC is sufficient for this deliberately small pilot.
    # Each positive/negative tie contributes half a win.
    positives = [s for y, s in zip(labels, scores) if y == 1]
    negatives = [s for y, s in zip(labels, scores) if y == 0]
    auc = (sum(float(p > n) + 0.5 * float(p == n) for p in positives for n in negatives)
           / (len(positives) * len(negatives))) if positives and negatives else None
    return {
        "scored_videos": len(labels), "real_scored": real_n, "fake_scored": fake_n,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "accuracy": (tp + tn) / len(labels) if labels else None,
        "balanced_accuracy": (specificity + recall) / 2 if real_n and fake_n else None,
        "real_specificity": specificity,
        "false_positive_rate": fp / real_n if real_n else None,
        "fake_precision": tp / (tp + fp) if tp + fp else None,
        "fake_recall": recall,
        "fake_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "roc_auc": auc,
        "always_fake_accuracy_on_same_scored_videos": fake_n / len(labels) if labels else None,
    }


def build_summary(records: list[dict], plan: list[dict], config_hash: str) -> dict:
    scored = [r for r in records if r["status"] == "scored"]
    counts = Counter(r["status"] for r in records)
    labels = [r["case"]["label"] for r in scored]
    scores = [r["prediction"]["fake_score"] for r in scored]
    methods = {}
    for method in METHODS:
        attempted = [r for r in records if r["case"]["method"] == method]
        successful = [r for r in scored if r["case"]["method"] == method]
        detected = sum(r["prediction"]["label"] == "Fake" for r in successful)
        methods[method] = {"planned": 5, "attempted": len(attempted),
                           "scored": len(successful), "correct_fake": detected,
                           "missed_as_real": len(successful) - detected,
                           "rejected_or_error": len(attempted) - len(successful),
                           "recall_on_scored": detected / len(successful) if successful else None}
    durations = [r["processing_seconds"] for r in scored]
    completed_ids = {r["case"]["case_id"] for r in records}
    return {
        "schema_version": 1, "evaluation_type": "small_fixed_validation_pilot",
        "complete": len(records) == len(plan), "config_sha256": config_hash,
        "seed": SEED, "threshold": 0.5, "planned_videos": len(plan),
        "attempted_videos": len(records), "scored_videos": len(scored),
        "insufficient_evidence": counts["insufficient_evidence"], "errors": counts["error"],
        "pending_case_ids": [r["case_id"] for r in plan if r["case_id"] not in completed_ids],
        "coverage_of_attempted": len(scored) / len(records) if records else None,
        "coverage_by_label": {name: {
            "planned": 20,
            "attempted": sum(r["case"]["label_name"] == name for r in records),
            "scored": sum(r["case"]["label_name"] == name for r in scored),
        } for name in ["Real", "Fake"]},
        "video_level_metrics_on_scored_only": binary_metrics(labels, scores),
        "fake_method_results": methods,
        "processing_seconds_on_scored": {
            "p50": float(np.percentile(durations, 50)) if durations else None,
            "p95": float(np.percentile(durations, 95)) if durations else None,
            "excludes": "model load, preflight hashing, report writing; not API latency",
        },
        "limitations": [
            "Small validation sample, not final test accuracy or a benchmark reproduction.",
            "10 source pairs; related original/manipulated clips are not independent observations.",
            "Metrics exclude rejected/error videos; inspect coverage and errors alongside metrics.",
            "YuNet preprocessing differs from the pretrained reference dlib pipeline.",
            "No threshold tuning, model training, or probability calibration performed.",
            "Checkpoint selection history must be considered before any independent-test claims.",
        ],
    }


def score_case(case: dict, data_root: Path, model, detector, config_hash: str) -> dict:
    start = time.perf_counter()
    report = {"case": case, "config_sha256": config_hash,
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "error", "prediction": None, "inference_performed": False}
    try:
        # Suppress the 16 routine frame messages, retaining one line per video.
        with redirect_stdout(io.StringIO()):
            crops, frames, _, video = preprocess_video(data_root / case["relative_path"], detector)
        prepared = time.perf_counter()
        scores = []
        if len(crops) >= MIN_FACES:
            with torch.inference_mode():
                for frame in frames:
                    if "crop_index" not in frame:
                        continue
                    tensor = torch.from_numpy(rgb_normalized_chw(crops[frame["crop_index"]])).unsqueeze(0)
                    logits = model(tensor)
                    if tuple(logits.shape) != (1, 2) or not torch.isfinite(logits).all().item():
                        raise ValueError("Invalid classifier logits.")
                    score = float(torch.softmax(logits, dim=1)[0, 1].item())
                    frame["fake_score"] = score
                    scores.append(score)
        prediction = aggregate_prediction(scores, threshold=0.5)
        report.update({"status": "scored" if prediction else "insufficient_evidence",
                       "prediction": prediction, "inference_performed": bool(scores),
                       "video": video, "frames": frames, "usable_faces": len(crops),
                       "preprocessing_seconds": prepared - start,
                       "inference_seconds": time.perf_counter() - prepared})
    except (ValueError, OSError, RuntimeError, cv2.error) as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    report["processing_seconds"] = time.perf_counter() - start
    return report


def write_outputs(run_dir: Path, records: list, plan: list, config_hash: str) -> dict:
    summary = build_summary(records, plan, config_hash)
    atomic_json(run_dir / "validation_summary.json", summary)
    with (run_dir / "video_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["case_id", "relative_path", "expected_label", "method", "status",
                  "prediction", "fake_score", "usable_faces", "processing_seconds", "error"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in records:
            case = report["case"]
            prediction = report.get("prediction") or {}
            writer.writerow({"case_id": case["case_id"], "relative_path": case["relative_path"],
                             "expected_label": case["label_name"], "method": case["method"],
                             "status": report["status"], "prediction": prediction.get("label", ""),
                             "fake_score": prediction.get("fake_score", ""),
                             "usable_faces": report.get("usable_faces", ""),
                             "processing_seconds": report["processing_seconds"],
                             "error": report.get("error", {}).get("message", "")})
    return summary


def run(args) -> int:
    data_root = args.data_root.expanduser().resolve()
    split_dir = args.split_dir or data_root / "FaceForensics++" / "splits"
    run_dir = args.run_dir.resolve()
    val_pairs, split_hashes = read_splits(split_dir)
    print("Official split pairs verified; source-ID overlap: 0", flush=True)
    selected = select_subset(val_pairs)
    print("Checking 40 selected files and hashing their contents...", flush=True)
    plan = preflight(data_root, selected)
    checkpoint = HERE.parent / "models" / "xception_best.pth"
    yunet = HERE.parent / "models" / "face_detection_yunet_2026may.onnx"
    verify_checkpoint(checkpoint)
    if not yunet.is_file() or sha256_file(yunet) != YUNET_SHA256:
        raise ValueError("Verified YuNet model is missing or different.")
    runtime = {"python": platform.python_version(), "torch": torch.__version__,
               "opencv": cv2.__version__, "numpy": np.__version__, "device": "cpu"}
    source_hashes = {name: sha256_file(HERE / name) for name in [
        "validate_subset.py", "predict_video.py", "video_preprocessing.py",
        "xception_loader.py", "xception_backbone.py"]}
    config = {"schema_version": 1, "data_root": str(data_root), "seed": SEED,
              "threshold": 0.5, "threads": args.threads, "batch_size": 1,
              "checkpoint_sha256": EXPECTED_SHA256, "yunet_sha256": YUNET_SHA256,
              "preprocessing_version": PREPROCESSING_VERSION, "source_sha256": source_hashes,
              "runtime": runtime, "official_split_pair_hashes": split_hashes, "plan": plan}
    config_hash = canonical_hash(config)
    config_path = run_dir / "run_config.json"
    if run_dir.exists() and any(run_dir.iterdir()) and not config_path.is_file():
        raise ValueError("Run directory is nonempty without its run_config.json. Use a new --run-dir.")
    if config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError("Inputs/code/settings differ from this saved run. Preserve it and use a "
                             "new --run-dir; results from different runs must not be mixed.")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(config_path, config)
    with (run_dir / "subset_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plan[0]))
        writer.writeheader()
        writer.writerows(plan)
    print("SUBSET PREFLIGHT: PASSED (40 present, nonempty, hashed; videos not yet decoded)", flush=True)
    print("20 Real + 20 Fake: Deepfakes 5 | Face2Face 5 | FaceSwap 5 | NeuralTextures 5", flush=True)
    print(f"Plan saved: {run_dir / 'subset_manifest.csv'}", flush=True)
    if args.prepare_only:
        print("No video inference performed. Run the same command without --prepare-only to start.")
        return 0
    report_dir = run_dir / "reports"
    report_dir.mkdir(exist_ok=True)
    records = []
    completed = {}
    for case in plan:
        path = report_dir / f"{case['case_id']}.json"
        if path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if (saved.get("config_sha256") != config_hash or saved.get("case") != case
                    or saved.get("status") not in {"scored", "insufficient_evidence", "error"}):
                raise ValueError(f"Inconsistent saved report: {path}. Preserve it for inspection.")
            completed[case["case_id"]] = saved
    records = [completed[c["case_id"]] for c in plan if c["case_id"] in completed]
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    model, detector = None, None
    if len(records) < len(plan):
        print(f"Loading models once; completed reports to reuse: {len(records)}/40", flush=True)
        model, model_metadata = load_xception(checkpoint)
        model_metadata.pop("preprocessing_implemented_in_this_pilot", None)
        model_metadata["batch_preprocessing_version"] = PREPROCESSING_VERSION
        detector = make_detector(yunet)
        atomic_json(run_dir / "model_metadata.json", model_metadata)
    interrupted = False
    try:
        for i, case in enumerate(plan, 1):
            if case["case_id"] in completed:
                continue
            report = score_case(case, data_root, model, detector, config_hash)
            atomic_json(report_dir / f"{case['case_id']}.json", report)
            completed[case["case_id"]] = report
            records = [completed[c["case_id"]] for c in plan if c["case_id"] in completed]
            write_outputs(run_dir, records, plan, config_hash)
            prediction = report.get("prediction")
            result = (f"{prediction['label']} | Fake score {prediction['fake_score']:.4f}"
                      if prediction else report["status"])
            print(f"[{i:02d}/40] {case['method']}/{Path(case['relative_path']).name} "
                  f"| Expected {case['label_name']} -> {result} "
                  f"| {report['processing_seconds']:.1f}s", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Completed per-video reports are saved; rerun the same command to resume.")
    summary = write_outputs(run_dir, records, plan, config_hash)
    metrics = summary["video_level_metrics_on_scored_only"]
    print(f"Complete: {summary['complete']} | Attempted: {len(records)}/40 "
          f"| Scored: {summary['scored_videos']} | Insufficient: {summary['insufficient_evidence']} "
          f"| Errors: {summary['errors']}")
    for key in ["accuracy", "balanced_accuracy", "false_positive_rate", "fake_precision",
                "fake_recall", "fake_f1", "roc_auc"]:
        print(f"{key}: {metrics[key]:.4f}" if metrics[key] is not None else f"{key}: undefined")
    print("These are small-validation-pilot metrics on scored videos; review coverage too.")
    print(f"Summary: {run_dir / 'validation_summary.json'}")
    print(f"Per-video results: {run_dir / 'video_results.csv'}")
    return 130 if interrupted else (3 if summary["errors"] else 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--run-dir", type=Path, default=HERE / "validation_runs" / "ffpp_val40_seed20261105")
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    try:
        return run(args)
    except (ValueError, OSError, RuntimeError, cv2.error) as error:
        print(f"VALIDATION STOPPED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
