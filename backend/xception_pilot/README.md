# DeepGuard: Xception loading pilot

This package downloads and verifies the official DeepfakeBench v1.0.1 Xception
detector checkpoint, loads every model parameter and buffer, and runs one CPU
forward pass on a synthetic tensor. It does not classify a video, measure
detection accuracy, or connect the model to FastAPI yet.

## Where to put this folder

Copy the entire `xception_pilot` folder into:

`C:\Projects\Deepfake_Detection_FYP\backend`

The resulting script path must be:

`C:\Projects\Deepfake_Detection_FYP\backend\xception_pilot\check_xception.py`

This is an additional folder. It does not replace `main.py`, `face_processing.py`,
the YuNet model, or existing dataset manifests.

## Run in the VS Code PowerShell terminal

Use the `.venv-model` environment already prepared for the model pilot:

```powershell
cd "C:\Projects\Deepfake_Detection_FYP\backend"
.\.venv-model\Scripts\python.exe .\xception_pilot\check_xception.py --download
```

The first run requires internet access and downloads approximately 87.8 MB to:

`C:\Projects\Deepfake_Detection_FYP\backend\models\xception_best.pth`

The script downloads from the official release, checks the full byte count and
SHA-256, and then installs the file. If an existing file passes verification,
it is reused. A mismatched existing file is reported rather than overwritten.
An interrupted network download can be retried by running the same command;
it starts a fresh download. No partial file is used as a model.

Expected final lines after a successful run include:

```text
Checkpoint SHA-256: PASS
Strict weight loading: PASS (283 state entries)
Class mapping: 0 = Real, 1 = Fake
Input shape: (1, 3, 256, 256)
Output shape: (1, 2)
CPU forward pass: PASS
MODEL LOAD TEST: PASSED
No video analysed. Detection accuracy has not been measured.
```

A fresh JSON report is saved in `xception_pilot\checks` on each successful run.
The synthetic test intentionally does not print Real/Fake scores because they
would not be meaningful predictions about a real video. The app's health
response may still say `model_loaded: false` until API integration is completed.

## What each Python file does

- `xception_backbone.py`: the matching architecture from the pinned release,
  with the benchmark registry import and decorator removed.
- `xception_loader.py`: checksum verification, optional download, strict CPU
  loading and checkpoint metadata. It loads the complete trained detector;
  no ImageNet download or randomly initialized classifier head is accepted.
- `check_xception.py`: command-line compatibility check and local report.

Only PyTorch and Python's standard library are used by this loading test.
The installed OpenCV, NumPy and Pillow packages are for subsequent preprocessing.

## Next integration step

Prepare model inputs from the original sampled frames using the selected
checkpoint's face alignment, RGB colour order, 256 x 256 size and normalisation
`(pixel / 255 - 0.5) / 0.5`. Reference resizing and face coverage also matter.
The existing unaligned 224 x 224 YuNet previews are not yet validated classifier
inputs. Start debugging on validation data; preserve the final test splits.

After preprocessing is verified, run actual video inference, choose and report
the video aggregation rule, evaluate it, then connect it to FastAPI and Flutter.
Frame score aggregation does not implement temporal/LSTM reasoning. Scores are
model outputs, not measured percentages of a video that were AI-generated.

## Source and checkpoint fingerprint

- Source repository: https://github.com/SCLBD/DeepfakeBench
- Release: https://github.com/SCLBD/DeepfakeBench/releases/tag/v1.0.1
- Source commit: `6bc3ea01cdc768b17bd37b5e0eca85ecdfb10472`
- Checkpoint: `xception_best.pth`
- Bytes: `87763531`
- SHA-256: `dd61186f9ac8e99aae85982cae338b555dfaeef042f6b699e5a55cae3998ba47`

The fingerprint was computed from the official release download on 2026-11-05.
It is a recorded checksum, not a publisher-signed checksum. The released
Xception configuration specifies FaceForensics++ c23 training, two classes,
256 x 256 inputs and means/standard deviations of 0.5.

Retain `ATTRIBUTION.md` and `LICENSE_DeepfakeBench.txt` with the adapted source.
This package does not contain the large checkpoint; the script fetches it from
the official source when requested.
