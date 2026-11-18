"""Build DeepGuard's dataset inventory without modifying any source video.

Run with Python 3.11 or newer; only the Python standard library is needed:
    python prepare_dataset_manifest.py --data-root "C:\\Users\\rajab\\DeepGuard_Data"

Expected folders below --data-root:
    FaceForensics++/splits/{train,val,test}.json
    FaceForensics++/original_sequences/youtube/c23/videos/
    FaceForensics++/manipulated_sequences/{Deepfakes,Face2Face,FaceSwap,NeuralTextures}/c23/videos/
    Celeb-DF-v2/{Celeb-real,YouTube-real,Celeb-synthesis}/
    Celeb-DF-v2/List_of_testing_videos.txt

Project labels: Real = 0, Fake = 1. Celeb-DF's text file uses the opposite
numeric convention; source_label preserves that value and label converts it.
All Celeb-DF videos outside the supplied test list remain reserved_unused.

Every run gets a separate directory under backend/manifests by default.
Inventory and issue reports are always written after metadata validation.
The four split manifests are written only when the inventory has no issues.
This checks filenames, split assignments, presence and byte sizes. It does NOT
decode videos, find duplicate video content, check face identities, extract
crops, train a classifier, or establish model accuracy.

FF++ source: https://github.com/ondyari/FaceForensics/tree/master/dataset/splits
Dataset documentation: https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md
Celeb-DF documentation: https://github.com/yuezunli/celeb-deepfakeforensics
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import stat
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from uuid import uuid4


VERSION = "1.0.0"
FFPP_METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
FFPP_ID_COUNTS = {"train": 720, "val": 140, "test": 140}
# Hashes of compact JSON for the official definitions verified on 2026-10-22.
# Whitespace and Windows line endings do not affect these semantic checks.
FFPP_DEFINITION_HASHES = {
    "train": "3859ca18f83935ceb183ae7a463bfaacd8b9e28b3d43fe0c86449c2679e84c5e",
    "val": "4636aa9dac20789696a053329dd0a9e576f2abf0dc5aea19d9e8b1dbcf0d7ea1",
    "test": "55ca26bd64610f2ab800819fce1c152aa2ed0b378b855f07f2685dcd88caace5",
}
CELEB_FOLDERS = {
    "Celeb-real": (590, 1, r"id\d+_\d{4}\.mp4"),
    "YouTube-real": (300, 1, r"\d{5}\.mp4"),
    "Celeb-synthesis": (5639, 0, r"id\d+_id\d+_\d{4}\.mp4"),
}
CELEB_TEST_COUNTS = {"Celeb-real": 108, "YouTube-real": 70, "Celeb-synthesis": 340}
COLUMNS = (
    "dataset", "category", "label_name", "label", "split", "group_id",
    "target_id", "source_id", "source_label", "relative_path", "file_path",
    "size_bytes", "file_status",
)


class DatasetError(ValueError):
    """An input layout or metadata file needs attention."""


def file_sha256(path: Path) -> str:
    """Used only on small metadata files, not the video dataset."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_ffpp_splits(root: Path) -> tuple[dict, dict, dict, list]:
    pairs_by_split, id_to_split, id_to_group, sources = {}, {}, {}, []
    for split, expected_ids in FFPP_ID_COUNTS.items():
        path = root / "FaceForensics++" / "splits" / f"{split}.json"
        pairs = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(pairs, list) or len(pairs) != expected_ids // 2:
            raise DatasetError(f"Unexpected pair count in {path}")
        for pair in pairs:
            if (not isinstance(pair, list) or len(pair) != 2
                    or any(not isinstance(x, str) or not re.fullmatch(r"\d{3}", x) for x in pair)
                    or pair[0] == pair[1]):
                raise DatasetError(f"Invalid ID pair in {path}: {pair!r}")
            group = "ffpp_" + "_".join(sorted(pair))
            for video_id in pair:
                if video_id in id_to_split:
                    raise DatasetError(f"Repeated FF++ ID {video_id}; split overlap or duplicate pair.")
                id_to_split[video_id] = split
                id_to_group[video_id] = group
        canonical = json.dumps(pairs, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        if digest != FFPP_DEFINITION_HASHES[split]:
            raise DatasetError(f"{path.name} differs from the verified official split definition.")
        pairs_by_split[split] = pairs
        sources.append({"relative_path": path.relative_to(root).as_posix(),
                        "sha256": file_sha256(path), "canonical_sha256": digest})
    if set(id_to_split) != {f"{i:03d}" for i in range(1000)}:
        raise DatasetError("FF++ split IDs must cover 000 through 999 exactly once.")
    return pairs_by_split, id_to_split, id_to_group, sources


def load_celeb_test(root: Path) -> tuple[dict, dict]:
    path = root / "Celeb-DF-v2" / "List_of_testing_videos.txt"
    entries, folder_counts = {}, Counter()
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2 or fields[0] not in {"0", "1"}:
            raise DatasetError(f"Invalid Celeb-DF test-list line {number}.")
        source_label, relative = int(fields[0]), fields[1]
        parts = relative.split("/")
        if len(parts) != 2 or parts[0] not in CELEB_FOLDERS:
            raise DatasetError(f"Invalid relative path on Celeb-DF test-list line {number}.")
        folder, filename = parts
        _, required_label, pattern = CELEB_FOLDERS[folder]
        if not re.fullmatch(pattern, filename) or source_label != required_label:
            raise DatasetError(f"Filename or label mismatch on Celeb-DF test-list line {number}.")
        if relative in entries:
            raise DatasetError(f"Duplicate Celeb-DF test entry: {relative}")
        entries[relative] = source_label
        folder_counts[folder] += 1
    if dict(folder_counts) != CELEB_TEST_COUNTS:
        raise DatasetError(f"Unexpected Celeb-DF test counts: {dict(folder_counts)}; expected {CELEB_TEST_COUNTS}.")
    return entries, {"relative_path": path.relative_to(root).as_posix(),
                     "sha256": file_sha256(path), "entries": len(entries),
                     "membership_source": "local dataset-supplied test list"}


def folder_files(root: Path, folder: Path, expected: int, counts: list, issues: list) -> dict:
    files = {}
    if folder.is_dir():
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() == ".mp4" and path.is_file():
                key = path.name.lower()
                if key in files:
                    raise DatasetError(f"Case-insensitive duplicate filename in {folder}: {path.name}")
                files[key] = path
    relative = folder.relative_to(root).as_posix()
    counts.append({"folder": relative, "expected_mp4_files": expected,
                   "found_mp4_files": len(files), "folder_exists": folder.is_dir()})
    if len(files) != expected:
        issues.append({"kind": "folder_count_mismatch", "relative_path": relative,
                       "detail": f"Expected {expected} MP4 files; found {len(files)}."})
    return files


def make_row(root: Path, path: Path, dataset: str, category: str, label: int,
             split: str, issues: list, group_id: str = "", target_id: str = "",
             source_id: str = "", source_label: int | str = "") -> dict:
    relative = path.relative_to(root).as_posix()
    size: int | str = ""
    try:
        info = path.stat()
        size = info.st_size
        if not stat.S_ISREG(info.st_mode):
            status = "not_a_file"
        else:
            status = "present_nonempty" if size > 0 else "empty"
    except FileNotFoundError:
        status = "missing"
    except OSError as exc:
        status = "stat_error"
        issues.append({"kind": "stat_error_detail", "relative_path": relative, "detail": str(exc)})
    if status != "present_nonempty":
        issues.append({"kind": status, "relative_path": relative, "detail": "Check this source file."})
    return {"dataset": dataset, "category": category, "label_name": "Fake" if label else "Real",
            "label": label, "split": split, "group_id": group_id, "target_id": target_id,
            "source_id": source_id, "source_label": source_label, "relative_path": relative,
            "file_path": str(path), "size_bytes": size, "file_status": status}


def collect_ffpp(root: Path, pairs_by_split: dict, id_to_split: dict,
                 id_to_group: dict, folder_counts: list, issues: list) -> list:
    rows = []
    base = root / "FaceForensics++"
    folders = {"original": base / "original_sequences" / "youtube" / "c23" / "videos"}
    folders.update({method: base / "manipulated_sequences" / method / "c23" / "videos"
                    for method in FFPP_METHODS})
    for category, folder in folders.items():
        available = folder_files(root, folder, 1000, folder_counts, issues)
        expected = []
        if category == "original":
            expected = [(f"{i}.mp4", id_to_split[i], id_to_group[i], i, "")
                        for i in sorted(id_to_split)]
        else:
            for split, pairs in pairs_by_split.items():
                for first, second in pairs:
                    # Both filename directions stay with the same official pair.
                    for target, source in ((first, second), (second, first)):
                        expected.append((f"{target}_{source}.mp4", split,
                                         id_to_group[target], target, source))
        label = 0 if category == "original" else 1
        for filename, split, group, target, source in expected:
            path = available.pop(filename.lower(), folder / filename)
            rows.append(make_row(root, path, "FaceForensics++", category, label, split,
                                 issues, group, target, source))
        for path in available.values():
            issues.append({"kind": "unexpected_ffpp_video", "relative_path": path.relative_to(root).as_posix(),
                           "detail": "Filename does not match an expected official ID pair."})
            rows.append(make_row(root, path, "FaceForensics++", category, label, "unassigned", issues))
    return rows


def collect_celeb(root: Path, test_entries: dict, folder_counts: list, issues: list) -> list:
    rows = []
    base = root / "Celeb-DF-v2"
    remaining = set(test_entries)
    for category, (expected_count, folder_source_label, pattern) in CELEB_FOLDERS.items():
        folder = base / category
        available = folder_files(root, folder, expected_count, folder_counts, issues)
        for filename, path in sorted(available.items()):
            if not re.fullmatch(pattern, filename):
                issues.append({"kind": "unexpected_celeb_filename", "relative_path": path.relative_to(root).as_posix(),
                               "detail": "Unexpected filename format for this Celeb-DF folder."})
            key = f"{category}/{filename}"
            is_test = key in test_entries
            remaining.discard(key)
            rows.append(make_row(root, path, "Celeb-DF-v2", category, 1 - folder_source_label,
                                 "external_test" if is_test else "reserved_unused", issues,
                                 source_label=test_entries[key] if is_test else ""))
    for key in sorted(remaining):
        folder, filename = key.split("/")
        source_label = test_entries[key]
        rows.append(make_row(root, base / folder / filename, "Celeb-DF-v2", folder,
                             1 - source_label, "external_test", issues, source_label=source_label))
    return rows


def write_csv(path: Path, rows: list, columns: tuple) -> None:
    # UTF-8 BOM makes Windows spreadsheet viewers display paths correctly.
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run(data_root: Path, output_root: Path) -> tuple[dict, Path]:
    root = data_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not root.is_dir():
        raise DatasetError(f"Data folder not found: {root}")
    if output_root == root or root in output_root.parents:
        raise DatasetError("Choose an output directory outside DeepGuard_Data to keep raw data separate.")

    pairs, id_splits, id_groups, ffpp_sources = load_ffpp_splits(root)
    test_entries, celeb_source = load_celeb_test(root)
    folder_counts, issues = [], []
    rows = collect_ffpp(root, pairs, id_splits, id_groups, folder_counts, issues)
    rows.extend(collect_celeb(root, test_entries, folder_counts, issues))
    rows.sort(key=lambda row: (row["dataset"], row["split"], row["relative_path"]))

    # Check assignments as well as definitions, including both fake-video IDs.
    for row in rows:
        if row["dataset"] != "FaceForensics++" or row["split"] == "unassigned":
            continue
        for column in ("target_id", "source_id"):
            video_id = row[column]
            if video_id and id_splits[video_id] != row["split"]:
                raise DatasetError(f"Source ID crosses splits: {row['relative_path']}")

    split_counts = []
    for dataset, split in [("FaceForensics++", s) for s in FFPP_ID_COUNTS] + [
        ("Celeb-DF-v2", "external_test"), ("Celeb-DF-v2", "reserved_unused")
    ]:
        selected = [r for r in rows if r["dataset"] == dataset and r["split"] == split]
        split_counts.append({"dataset": dataset, "split": split, "listed": len(selected),
                             "real": sum(r["label"] == 0 for r in selected),
                             "fake": sum(r["label"] == 1 for r in selected),
                             "present_nonempty": sum(r["file_status"] == "present_nonempty" for r in selected)})
    summary = {
        "script_version": VERSION, "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "data_root": str(root), "status": "inventory_passed" if not issues else "needs_attention",
        "label_mapping": {"0": "Real", "1": "Fake"},
        "celeb_source_label_mapping": {"0": "Fake", "1": "Real"},
        "celeb_label_conversion": "label = 1 - source_label for test-list entries",
        "ffpp_official_definitions_match": True, "ffpp_source_id_overlap": 0,
        "ffpp_original_id_counts": FFPP_ID_COUNTS, "folder_counts": folder_counts,
        "split_counts": split_counts, "file_status_counts": dict(Counter(r["file_status"] for r in rows)),
        "issue_count": len(issues), "issue_counts_by_kind": dict(Counter(i["kind"] for i in issues)),
        "metadata_sources": ffpp_sources + [celeb_source],
        "checks_not_performed": ["video_decoding", "duplicate_video_content", "cross_dataset_content_overlap",
                                 "face_identity_disjointness", "face_crop_quality", "model_training_or_evaluation"],
        "next_stage": "Decode audit and pilot preprocessing; inventory success is not proof that all videos are valid.",
        "manifests_written": [],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"inventory_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
    output.mkdir()
    write_csv(output / "file_inventory.csv", rows, COLUMNS)
    write_csv(output / "issues.csv", issues, ("kind", "relative_path", "detail"))
    if not issues:
        for split in FFPP_ID_COUNTS:
            name = f"ffpp_{split}.csv"
            selected = [r for r in rows if r["dataset"] == "FaceForensics++" and r["split"] == split]
            write_csv(output / name, selected, COLUMNS)
            summary["manifests_written"].append(name)
        name = "celebdf_external_test.csv"
        write_csv(output / name, [r for r in rows if r["dataset"] == "Celeb-DF-v2"
                                 and r["split"] == "external_test"], COLUMNS)
        summary["manifests_written"].append(name)
    (output / "inventory_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "manifests")
    args = parser.parse_args()
    try:
        summary, output = run(args.data_root, args.output_dir)
    except (DatasetError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"Inventory failed: {exc}", file=sys.stderr)
        return 1
    print("DeepGuard dataset inventory")
    print("Project labels: 0 = Real, 1 = Fake")
    print("FF++ official split IDs: train 720 | val 140 | test 140 | overlap 0")
    print("\nFolder counts (found / expected):")
    for item in summary["folder_counts"]:
        print(f"  {item['found_mp4_files']:5d} / {item['expected_mp4_files']:5d}  {item['folder']}")
    print("\nSplit inventory:")
    for item in summary["split_counts"]:
        print(f"  {item['dataset']} {item['split']}: {item['listed']} listed "
              f"({item['real']} real, {item['fake']} fake), {item['present_nonempty']} present and nonempty")
    print(f"\nStatus: {summary['status']}")
    print(f"Issues: {summary['issue_count']}")
    if summary["issue_count"]:
        print("Split manifests were withheld. See issues.csv, fix the source issue, and run again.")
        print(json.dumps(summary["issue_counts_by_kind"], indent=2))
    else:
        print("Split manifests saved. Videos have NOT been decoded or checked for duplicate content.")
    print(f"Reports saved:\n{output}")
    return 0 if not summary["issue_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
