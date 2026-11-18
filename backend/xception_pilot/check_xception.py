r"""Download, strictly load and exercise the official Xception weights on CPU.

Usage from the backend folder in Windows PowerShell:
    .\.venv-model\Scripts\python.exe .\xception_pilot\check_xception.py --download

This is a synthetic-input compatibility test, not an actual video prediction.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
import time
from uuid import uuid4

import torch

if __package__:
    from .xception_loader import CHECKPOINT_NAME, ensure_checkpoint, load_xception
else:
    from xception_loader import CHECKPOINT_NAME, ensure_checkpoint, load_xception


def main() -> int:
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="DeepGuard Xception CPU loading test")
    parser.add_argument("--download", action="store_true", help="Download the official weights if missing")
    parser.add_argument("--checkpoint", type=Path,
                        default=directory.parent / "models" / CHECKPOINT_NAME)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be a positive integer")
    torch.set_num_threads(args.threads)
    print("DeepGuard Xception model loading test", flush=True)
    print(f"PyTorch: {torch.__version__} | Device: cpu | Threads: {args.threads}", flush=True)
    print("Input: synthetic tensor, not a video.", flush=True)
    try:
        checkpoint = args.checkpoint.expanduser().resolve()
        ensure_checkpoint(checkpoint, allow_download=args.download)
        started = time.perf_counter()
        model, metadata = load_xception(checkpoint)
        load_seconds = time.perf_counter() - started
        print("Checkpoint SHA-256: PASS", flush=True)
        print(f"Strict weight loading: PASS ({metadata['state_entries_loaded']} state entries)", flush=True)
        print("Class mapping: 0 = Real, 1 = Fake", flush=True)
        test_input = torch.zeros((1, 3, 256, 256), dtype=torch.float32, device="cpu")
        started = time.perf_counter()
        with torch.inference_mode():
            logits = model(test_input)
            if tuple(logits.shape) != (1, 2):
                raise ValueError(f"Unexpected output shape: {tuple(logits.shape)}")
            if not torch.isfinite(logits).all().item():
                raise ValueError("Forward pass produced non-finite logits.")
            probabilities = torch.softmax(logits, dim=1)
            if not torch.isfinite(probabilities).all().item():
                raise ValueError("Forward pass produced non-finite scores.")
            if not torch.allclose(probabilities.sum(dim=1), torch.ones(1), atol=1e-6):
                raise ValueError("Binary scores did not sum to one.")
        forward_seconds = time.perf_counter() - started
        print("Input shape: (1, 3, 256, 256)", flush=True)
        print("Output shape: (1, 2)", flush=True)
        print("CPU forward pass: PASS", flush=True)

        now = datetime.now().astimezone()
        report = {
            "status": "model_load_test_passed",
            "created_at": now.isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": str(torch.__version__),
            "torch_distribution_version": importlib.metadata.version("torch"),
            "device": "cpu",
            "cpu_threads": args.threads,
            "input_is_synthetic": True,
            "input_shape": list(test_input.shape),
            "output_shape": list(logits.shape),
            "finite_outputs": True,
            "binary_score_sum_checked": True,
            "model_loading_seconds": round(load_seconds, 4),
            "synthetic_forward_seconds": round(forward_seconds, 4),
            "video_analysis_performed": False,
            "accuracy_evaluated": False,
            "api_integrated": False,
            "model": metadata,
        }
        reports_directory = directory / "checks"
        reports_directory.mkdir(exist_ok=True)
        report_path = reports_directory / (
            f"xception_load_{now:%Y%m%d_%H%M%S}_{uuid4().hex[:6]}.json"
        )
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("MODEL LOAD TEST: PASSED", flush=True)
        print("No video analysed. Detection accuracy has not been measured.", flush=True)
        print(f"Report saved:\n{report_path}", flush=True)
        return 0
    except Exception as exc:
        print(f"MODEL LOAD TEST: FAILED\n{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        print("Copy the complete terminal output for troubleshooting.", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
