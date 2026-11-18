"""Strict CPU loader for one published DeepfakeBench detector checkpoint.

This module does not preprocess videos or calibrate scores. The model expects
inputs matching the reference preprocessing, which is a separate integration step.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path
import os
import tempfile
from urllib.request import Request, urlopen

import torch
from torch import nn

if __package__:
    from .xception_backbone import Xception
else:
    from xception_backbone import Xception


MODEL_ID = "deepfakebench-xception-v1.0.1"
CHECKPOINT_NAME = "xception_best.pth"
CHECKPOINT_URL = (
    "https://github.com/SCLBD/DeepfakeBench/releases/download/"
    "v1.0.1/xception_best.pth"
)
EXPECTED_BYTES = 87_763_531
# Independently computed from the official release download on 2026-11-05.
# This is our recorded file fingerprint, not a publisher-signed checksum.
EXPECTED_SHA256 = "dd61186f9ac8e99aae85982cae338b555dfaeef042f6b699e5a55cae3998ba47"
UPSTREAM_COMMIT = "6bc3ea01cdc768b17bd37b5e0eca85ecdfb10472"
BACKBONE_CONFIG = {"mode": "original", "num_classes": 2, "inc": 3, "dropout": False}
LABELS = {0: "Real", 1: "Fake"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"Checkpoint file not found: {path}. Run with --download.")
    actual_size = path.stat().st_size
    if actual_size != EXPECTED_BYTES:
        raise ValueError(
            f"Checkpoint size mismatch: got {actual_size:,} bytes; expected "
            f"{EXPECTED_BYTES:,}. Keep the existing file for inspection and report this error."
        )
    digest = sha256_file(path)
    if digest != EXPECTED_SHA256:
        raise ValueError(
            "Checkpoint SHA-256 mismatch. This loader accepts only the verified "
            "v1.0.1 release file; do not bypass this check."
        )
    return digest


def ensure_checkpoint(path: Path, allow_download: bool = False) -> None:
    """Download a missing checkpoint to a temporary file, verify, then install it."""
    if path.exists():
        verify_checkpoint(path)
        print("Existing checkpoint verified; download skipped.", flush=True)
        return
    if not allow_download:
        raise ValueError(f"Checkpoint not found: {path}. Run this command with --download.")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".part", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    print(f"Downloading official checkpoint ({EXPECTED_BYTES / 1_000_000:.1f} MB)...", flush=True)
    request = Request(CHECKPOINT_URL, headers={"User-Agent": "DeepGuard-FYP-model-pilot/1.0"})
    received = 0
    last_percent = -10
    try:
        with os.fdopen(descriptor, "wb") as handle:
            with urlopen(request, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > EXPECTED_BYTES:
                        raise ValueError("Download is larger than the expected release file.")
                    handle.write(chunk)
                    percent = received * 100 // EXPECTED_BYTES
                    if percent >= last_percent + 10 or received == EXPECTED_BYTES:
                        print(f"  Download: {percent}%", flush=True)
                        last_percent = percent
        verify_checkpoint(temporary_path)
        # Do not replace an existing checkpoint, including one created by another run.
        if path.exists():
            raise ValueError("Checkpoint destination appeared during download. Run the check again.")
        temporary_path.rename(path)
        print("Download verified and saved.", flush=True)
    finally:
        temporary_path.unlink(missing_ok=True)


class XceptionForInference(nn.Module):
    """Preserve the detector's backbone.* state-dictionary names and logits."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = Xception(dict(BACKBONE_CONFIG))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone.features(images)
        return self.backbone.classifier(features)


def load_xception(checkpoint: Path) -> tuple[XceptionForInference, dict]:
    """Require every expected parameter and buffer; never use a random head."""
    checkpoint = Path(checkpoint).resolve()
    digest = verify_checkpoint(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Expected a nonempty model state dictionary.")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor)
               for key, value in state.items()):
        raise ValueError("Checkpoint must contain tensor state entries only.")

    model = XceptionForInference()
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing or unexpected:
        raise ValueError(f"State keys do not match: missing={missing[:8]}, unexpected={unexpected[:8]}")
    for name, tensor in state.items():
        if tensor.shape != expected[name].shape:
            raise ValueError(f"Wrong tensor shape for {name}: {tuple(tensor.shape)}")
        if tensor.dtype != expected[name].dtype:
            raise ValueError(f"Wrong tensor dtype for {name}: {tensor.dtype}")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            raise ValueError(f"Non-finite checkpoint values in {name}.")
    model.load_state_dict(state, strict=True)
    model.cpu().eval()
    metadata = {
        "model_id": MODEL_ID,
        "checkpoint_path": str(checkpoint),
        "checkpoint_source": CHECKPOINT_URL,
        "checkpoint_sha256": digest,
        "checkpoint_bytes": EXPECTED_BYTES,
        "upstream_commit": UPSTREAM_COMMIT,
        "strict_loading": True,
        "state_entries_loaded": len(state),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "label_mapping": LABELS,
        "reference_input_size": [256, 256],
        "reference_colour_order": "RGB",
        "reference_mean": [0.5, 0.5, 0.5],
        "reference_std": [0.5, 0.5, 0.5],
        "reference_face_alignment_required": True,
        "preprocessing_implemented_in_this_pilot": False,
    }
    return model, metadata
