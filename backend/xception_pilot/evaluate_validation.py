"""Resume a fixed 200-video FF++ validation study using the app's XceptionEngine.

Includes the original 40-video plan plus 160 additional videos. It does not
train, tune a threshold, use test videos, or modify the application.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager, redirect_stdout
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
SEED = 20261109
METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
RUN_NAME = "ffpp_val200_seed20261109"
SOURCE_FILES = (
    "main.py", "inference_service.py", "api_contract.py",
    "xception_pilot/evaluate_validation.py", "xception_pilot/validate_subset.py",
    "xception_pilot/video_preprocessing.py", "xception_pilot/predict_video.py",
    "xception_pilot/xception_loader.py", "xception_pilot/xception_backbone.py",
)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@contextmanager
def run_lock(directory):
    """OS releases this advisory lock on exit/crash; no stale-PID deletion needed."""
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / ".evaluation.lock").open("a+b")
    if handle.seek(0, os.SEEK_END) == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    locked = False
    try:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as error:
            raise ValueError("Another evaluation is using this run directory. Run one instance only.") from error
        yield
    finally:
        if locked:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def make_plan(val_pairs, pilot40):
    """Extend the prior plan, without selecting examples based on predictions."""
    rows = [{k: v for k, v in row.items() if k != "case_id"} | {"cohort": "pilot40"}
            for row in pilot40]
    used = {row["source_pair"] for row in rows}
    available = [sorted(pair) for pair in val_pairs
                 if "_".join(sorted(pair)) not in used]
    if len(rows) != 40 or len(used) != 10 or len(available) != 60:
        raise ValueError("The original 40-video selection differs from the expected 10-pair plan.")
    rng = random.Random(SEED)
    selected = rng.sample(sorted(available), 40)
    methods = list(METHODS) * 20
    rng.shuffle(methods)
    slot = 0
    for a, b in selected:
        for first, second in ((a, b), (b, a)):
            common = {"source_pair": f"{a}_{b}", "split": "val", "cohort": "additional160"}
            rows.append(common | {"relative_path":
                f"FaceForensics++/original_sequences/youtube/c23/videos/{first}.mp4",
                "label": 0, "label_name": "Real", "method": "Original"})
            method = methods[slot]
            slot += 1
            rows.append(common | {"relative_path":
                f"FaceForensics++/manipulated_sequences/{method}/c23/videos/{first}_{second}.mp4",
                "label": 1, "label_name": "Fake", "method": method})
    rng.shuffle(rows)
    expected = {"Original": 100, **{method: 25 for method in METHODS}}
    if (Counter(row["method"] for row in rows) != expected
            or len({row["relative_path"] for row in rows}) != 200
            or len({row["source_pair"] for row in rows}) != 50):
        raise ValueError("Unexpected validation plan composition.")
    for index, row in enumerate(rows, 1):
        row["case_id"] = f"case_{index:03d}"
    return rows


def check_inputs(data_root, rows):
    plan, by_hash = [], defaultdict(list)
    for row in rows:
        path = data_root / row["relative_path"]
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty selected video: {path}. No substitution was made.")
        case = row | {"size_bytes": path.stat().st_size, "sha256": file_hash(path)}
        plan.append(case)
        by_hash[case["sha256"]].append(case)
    duplicates = []
    for digest, cases in by_hash.items():
        if len(cases) < 2:
            continue
        if len({case["label"] for case in cases}) != 1:
            names = ", ".join(case["relative_path"] for case in cases)
            raise ValueError(f"Identical selected file bytes have conflicting Real/Fake labels: {names}")
        duplicates.append({"sha256": digest, "paths": [case["relative_path"] for case in cases]})
    return plan, duplicates


def bind_config(directory, config):
    path = directory / "run_config.json"
    if path.exists():
        if read_json(path) != config:
            raise ValueError("Saved inputs/code/models/runtime differ. Keep this evidence and use a new --run-dir; do not mix runs.")
    else:
        unexpected = [p.name for p in directory.iterdir()
                      if p.name not in {".evaluation.lock", "run_config.json.tmp"}]
        if unexpected:
            raise ValueError("Output directory already contains other files. Use a new --run-dir.")
        atomic_json(path, config)
    return fingerprint(config)


def validate_result(result, config):
    """Reject malformed outputs or a changed decision rule before counting them."""
    if not isinstance(result, dict):
        raise ValueError("Engine did not return an object.")
    model = result.get("model", {})
    prep = result.get("preprocessing", {})
    if (model.get("checkpoint_sha256") != config["checkpoint_sha256"]
            or prep.get("version") != config["preprocessing_version"]
            or prep.get("target_frames") != 16
            or prep.get("minimum_usable_faces") != 8):
        raise ValueError("Engine provenance or face policy differs from this evaluation.")
    sampled, usable = prep.get("sampled_frames"), prep.get("usable_faces")
    if (type(sampled) is not int or type(usable) is not int
            or not 0 <= usable <= sampled <= 16):
        raise ValueError("Invalid sampled/usable frame counts.")
    for key in ("preprocessing", "inference", "processing"):
        seconds = result.get("timing_seconds", {}).get(key)
        if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or seconds < 0):
            raise ValueError("Invalid processing time.")
    frames = result.get("frame_scores")
    if not isinstance(frames, list):
        raise ValueError("Missing frame scores.")
    if result.get("status") == "insufficient_evidence":
        if (result.get("prediction") is not None or result.get("analysis_performed") is not False
                or usable >= 8 or frames):
            raise ValueError("Inconsistent insufficient-evidence result.")
        return
    prediction = result.get("prediction")
    if (result.get("status") != "analysed" or result.get("analysis_performed") is not True
            or not isinstance(prediction, dict) or not 8 <= len(frames) == usable <= 16):
        raise ValueError("Inconsistent analysed result.")
    scores, indices = [], []
    for frame in frames:
        score, index = frame.get("fake_score"), frame.get("frame_index")
        if (isinstance(score, bool) or not isinstance(score, (float, int))
                or not math.isfinite(score) or not 0 <= score <= 1
                or type(index) is not int or index < 0):
            raise ValueError("Invalid per-frame score or index.")
        scores.append(score)
        indices.append(index)
    if len(set(indices)) != len(indices):
        raise ValueError("Repeated scored frame indices.")
    mean = sum(scores) / len(scores)
    label_id = int(mean >= 0.5)
    if (prediction.get("decision_threshold") != 0.5
            or prediction.get("class_id") != label_id
            or prediction.get("label") != ("Fake" if label_id else "Real")
            or prediction.get("frames_scored") != len(scores)
            or prediction.get("calibrated") is not False
            or prediction.get("aggregation") != "mean_frame_softmax_fake_score"
            or not math.isclose(prediction.get("fake_score", -1), mean, rel_tol=0, abs_tol=1e-8)):
        raise ValueError("Prediction is inconsistent with the frozen 0.50 rule and mean frame scores.")


def save_record(directory, record):
    record = dict(record)
    record["record_sha256"] = fingerprint(record)
    atomic_json(directory / "reports" / f"{record['case']['case_id']}.json", record)


def load_records(directory, plan, config_hash, config):
    records = []
    for case in plan:
        path = directory / "reports" / f"{case['case_id']}.json"
        if not path.exists():
            continue
        record = read_json(path)
        digest = record.pop("record_sha256", None)
        if (digest != fingerprint(record) or record.get("config_sha256") != config_hash
                or record.get("case") != case
                or record.get("status") not in {"scored", "insufficient_evidence", "error"}):
            raise ValueError(f"Saved record is inconsistent: {path}. Preserve it for review.")
        if record["status"] == "error":
            if record.get("result") is not None or not isinstance(record.get("error"), dict):
                raise ValueError(f"Invalid error record: {path}")
        else:
            validate_result(record["result"], config)
            expected = "scored" if record["result"]["status"] == "analysed" else "insufficient_evidence"
            if record["status"] != expected:
                raise ValueError(f"Saved status differs from its result: {path}")
        records.append(record)
    return records


def percentile(values, percentage):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def group_summary(records, planned, metrics_fn):
    scored = [record for record in records if record["status"] == "scored"]
    counts = Counter(record["status"] for record in records)
    metrics = metrics_fn([r["case"]["label"] for r in scored],
                         [r["result"]["prediction"]["fake_score"] for r in scored])
    return {
        "planned": len(planned), "attempted": len(records), "scored": len(scored),
        "insufficient_evidence": counts["insufficient_evidence"], "errors": counts["error"],
        "coverage_of_attempted": len(scored) / len(records) if records else None,
        "metrics_on_scored_only": metrics,
    }


def build_summary(records, plan, config_hash, metrics_fn):
    root = group_summary(records, plan, metrics_fn)
    scored = [r for r in records if r["status"] == "scored"]
    durations = [r["result"]["timing_seconds"]["processing"] for r in scored]
    by_category = {name: group_summary([r for r in records if r["case"]["method"] == name],
                                     [c for c in plan if c["method"] == name], metrics_fn)
                   for name in ("Original", *METHODS)}
    cohorts = {name: group_summary([r for r in records if r["case"]["cohort"] == name],
                                  [c for c in plan if c["cohort"] == name], metrics_fn)
               for name in ("pilot40", "additional160")}
    return root | {
        "schema_version": 1, "evaluation_type": "ffpp_c23_validation_200",
        "complete": len(records) == len(plan), "provisional": len(records) != len(plan),
        "config_sha256": config_hash, "threshold": 0.5, "seed": SEED,
        "planned_source_pairs": len({c["source_pair"] for c in plan}),
        "coverage_by_label": {name: group_summary(
            [r for r in records if r["case"]["label_name"] == name],
            [c for c in plan if c["label_name"] == name], metrics_fn)
            for name in ("Real", "Fake")},
        "by_category": by_category, "by_cohort": cohorts,
        "processing_seconds_on_scored": {
            "p50": percentile(durations, 50), "p95": percentile(durations, 95),
            "mean": sum(durations) / len(durations) if durations else None,
            "excludes": "network/upload, API validation, model loading, hashing and report writing",
        },
        "pending_videos": len(plan) - len(records),
        "limitations": [
            "Development validation, not final test accuracy or performance on internet videos.",
            "Includes 40 earlier pilot cases plus 160 additions; these are not 240 distinct cases.",
            "Related videos belong to 50 source pairs; frames and related clips are not independent samples.",
            "Balanced Real/Fake selection changes class prevalence; precision is specific to this sample.",
            "Scored-only metrics exclude abstentions/errors; inspect coverage separately.",
            "Same application inference engine, but not an HTTP/mobile end-to-end test.",
            "HTTP 30-second/30-MB limits are not applied; engine limits (120 seconds, 200 MiB, 50000 frames, 4K pixels) still apply.",
            "YuNet alignment differs from the checkpoint reference dlib preprocessing.",
            "No training, tuning, or probability calibration. Pretrained checkpoint selection history remains a limitation.",
            "Byte hashes cover the selected files only, not semantic duplicates or cross-dataset identity overlap.",
        ],
    }


def write_outputs(directory, records, plan, config_hash, metrics_fn):
    summary = build_summary(records, plan, config_hash, metrics_fn)
    atomic_json(directory / "validation_summary.json", summary)
    fields = ["case_id", "relative_path", "sha256", "source_pair", "cohort", "expected_label",
              "method", "status", "prediction", "fake_score", "sampled_frames", "usable_faces",
              "processing_seconds", "error"]
    temporary = directory / "video_results.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            case, result = record["case"], record.get("result") or {}
            prediction, prep = result.get("prediction") or {}, result.get("preprocessing") or {}
            writer.writerow({
                **{key: case[key] for key in ("case_id", "relative_path", "sha256", "source_pair", "cohort", "method")},
                "expected_label": case["label_name"], "status": record["status"],
                "prediction": prediction.get("label", ""), "fake_score": prediction.get("fake_score", ""),
                "sampled_frames": prep.get("sampled_frames", ""), "usable_faces": prep.get("usable_faces", ""),
                "processing_seconds": result.get("timing_seconds", {}).get("processing", ""),
                "error": record.get("error", {}).get("message", ""),
            })
    os.replace(temporary, directory / "video_results.csv")
    return summary


def execute(directory, data_root, plan, config, metrics_fn, engine_factory, max_new=None):
    config_hash = fingerprint(config)
    (directory / "reports").mkdir(exist_ok=True)
    records = load_records(directory, plan, config_hash, config)
    done = {r["case"]["case_id"] for r in records}
    print(f"Saved outcomes to reuse: {len(done)}/{len(plan)}", flush=True)
    engine = engine_factory() if len(done) < len(plan) else None
    new_count, interrupted = 0, False
    try:
        for case in plan:
            if case["case_id"] in done:
                continue
            if max_new is not None and new_count >= max_new:
                break
            path = data_root / case["relative_path"]
            if not path.is_file() or path.stat().st_size != case["size_bytes"] or file_hash(path) != case["sha256"]:
                raise ValueError(f"Video changed after preflight: {path}. Run stopped without replacing it.")
            record = {"case": case, "config_sha256": config_hash,
                      "created_at_utc": datetime.now(timezone.utc).isoformat(), "result": None}
            started = time.perf_counter()
            try:
                with redirect_stdout(io.StringIO()):
                    result = engine.analyse(path)  # No label or expected result is passed to the model.
                validate_result(result, config)
                record.update(result=result, status="scored" if result["status"] == "analysed" else "insufficient_evidence")
            except Exception as error:
                record.update(status="error", error={"type": type(error).__name__, "message": str(error)})
            record["wall_seconds"] = time.perf_counter() - started
            save_record(directory, record)
            done.add(case["case_id"])
            new_count += 1
            records.append(record)
            write_outputs(directory, records, plan, config_hash, metrics_fn)
            prediction = (record.get("result") or {}).get("prediction")
            outcome = (f"{prediction['label']} | Fake score {prediction['fake_score']:.4f}"
                       if prediction else record["status"])
            print(f"[{len(done):03d}/{len(plan)}] {case['method']}/{path.name} "
                  f"| Expected {case['label_name']} -> {outcome} | {record['wall_seconds']:.1f}s", flush=True)
            if record["status"] == "error":
                print(f"  {record['error']['type']}: {record['error']['message']}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Completed videos are saved; the unfinished video will restart.", flush=True)
    finally:
        # Disk is authoritative even if interruption occurred just after committing a record.
        records = load_records(directory, plan, config_hash, config)
        summary = write_outputs(directory, records, plan, config_hash, metrics_fn)
    print(f"Complete: {summary['complete']} | Attempted: {summary['attempted']}/{summary['planned']} "
          f"| Scored: {summary['scored']} | Insufficient: {summary['insufficient_evidence']} | Errors: {summary['errors']}")
    print("Validation metrics on scored videos" + (" (PROVISIONAL: run incomplete):" if not summary["complete"] else ":"))
    for key in ("accuracy", "balanced_accuracy", "false_positive_rate", "fake_precision", "fake_recall", "fake_f1", "roc_auc"):
        value = summary["metrics_on_scored_only"][key]
        print(f"  {key}: {value:.4f}" if value is not None else f"  {key}: undefined")
    print(f"Summary: {directory / 'validation_summary.json'}")
    print(f"Per-video results: {directory / 'video_results.csv'}")
    if not summary["complete"]:
        print("Resume with the same command; completed outcomes will be reused.")
    return 130 if interrupted else (3 if summary["errors"] else 0)


def run(args):
    sys.path.insert(0, str(BACKEND))
    import cv2
    import numpy as np
    import torch
    from inference_service import XceptionEngine
    from xception_pilot.validate_subset import read_splits, select_subset, binary_metrics
    from xception_pilot.video_preprocessing import PREPROCESSING_VERSION, YUNET_SHA256
    from xception_pilot.xception_loader import verify_checkpoint, EXPECTED_SHA256

    data_root = args.data_root.expanduser().resolve()
    run_dir = (args.run_dir or HERE / "validation_runs" / RUN_NAME).resolve()
    with run_lock(run_dir):
        pairs, split_hashes = read_splits(data_root / "FaceForensics++" / "splits")
        print("Official train/val/test split pairs verified; source-ID overlap: 0.", flush=True)
        rows = make_plan(pairs, select_subset(pairs))
        print("Checking and hashing 200 selected VALIDATION files...", flush=True)
        plan, duplicates = check_inputs(data_root, rows)
        checkpoint = BACKEND / "models" / "xception_best.pth"
        verify_checkpoint(checkpoint)
        if file_hash(BACKEND / "models" / "face_detection_yunet_2026may.onnx") != YUNET_SHA256:
            raise ValueError("YuNet model fingerprint differs; keep the file for inspection.")
        config = {
            "schema_version": 1, "protocol": RUN_NAME, "data_root": str(data_root), "seed": SEED,
            "threshold": 0.5, "threads": 2, "batch_size": 1, "device": "cpu",
            "checkpoint_sha256": EXPECTED_SHA256, "yunet_sha256": YUNET_SHA256,
            "preprocessing_version": PREPROCESSING_VERSION, "official_split_pair_hashes": split_hashes,
            "source_sha256": {name: file_hash(BACKEND / name) for name in SOURCE_FILES},
            "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                        "opencv": cv2.__version__, "numpy": np.__version__, "platform": platform.platform()},
            "same_label_byte_duplicate_groups": duplicates, "plan": plan,
        }
        config_hash = bind_config(run_dir, config)
        with (run_dir / "subset_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(plan[0]))
            writer.writeheader()
            writer.writerows(plan)
        print("EVALUATION PREFLIGHT: PASSED (200 present, nonempty and hashed; videos not yet decoded)")
        print("100 Real + 100 Fake: Deepfakes 25 | Face2Face 25 | FaceSwap 25 | NeuralTextures 25")
        print("50 source pairs. Includes the original 40 cases plus 160 additions.")
        print(f"Selected same-label byte-duplicate groups: {len(duplicates)}")
        print(f"Configuration SHA-256: {config_hash}")
        print(f"Output: {run_dir}")
        if args.prepare_only:
            print("No inference performed. Remove --prepare-only to run; add --max-new-videos 50 for a batch.")
            return 0
        return execute(run_dir, data_root, plan, config, binary_metrics, XceptionEngine, args.max_new_videos)


def positive_integer(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Use a positive integer.")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, help="New directory only when intentionally starting a separate experiment.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-new-videos", type=positive_integer,
                        help="Stop after this many additional cases; next invocation resumes.")
    args = parser.parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Stopped during setup. Rerun the same command to resume.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"EVALUATION STOPPED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
