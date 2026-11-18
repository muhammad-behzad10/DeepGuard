# DeepGuard Xception video pilot

This add-on runs a local MP4 through YuNet alignment and the already verified
DeepfakeBench Xception checkpoint. It is a validation/integration pilot, not a
measured accuracy claim. It does not change the running FastAPI or Flutter app.

## Install on the laptop

The original Xception loading test must have passed first. Extract this ZIP
somewhere temporary. Copy these three files from its `xception_pilot` folder
into `C:\Projects\Deepfake_Detection_FYP\backend\xception_pilot`:

- `video_preprocessing.py`
- `predict_video.py`
- `VIDEO_PILOT_README.md`

They sit beside `check_xception.py`, `xception_loader.py`, and
`xception_backbone.py`. Do not create a second nested `xception_pilot` folder.
Keep the original `ATTRIBUTION.md` and `LICENSE_DeepfakeBench.txt`.

Use the existing `.venv-model` environment, which already has PyTorch, NumPy
and OpenCV. No additional packages or model download are required for this
add-on. The terminal showing `(.venv)` is harmless: each command below explicitly
uses `.venv-model\Scripts\python.exe`.

Default existing models:

```text
C:\Projects\Deepfake_Detection_FYP\backend\models\xception_best.pth
C:\Projects\Deepfake_Detection_FYP\backend\models\face_detection_yunet_2026may.onnx
```

Both model fingerprints are verified. The script stops on a missing or changed
model. It never downloads or overwrites models automatically.

## Run two development examples

Open VS Code and a PowerShell terminal. FastAPI, Flutter, and the mobile USB
connection are not required for this standalone step. These example IDs come
from the first pair (`720`, `672`) of the official FF++ validation split; do not
switch to final test examples to tune the pipeline.

First run the original video:

```powershell
cd "C:\Projects\Deepfake_Detection_FYP\backend"

$videoPath = "C:\Users\rajab\DeepGuard_Data\FaceForensics++\original_sequences\youtube\c23\videos\720.mp4"

.\.venv-model\Scripts\python.exe .\xception_pilot\predict_video.py --video "$videoPath" --expected-label Real
```

Wait for completion, then run its manipulated example:

```powershell
$videoPath = "C:\Users\rajab\DeepGuard_Data\FaceForensics++\manipulated_sequences\Deepfakes\c23\videos\720_672.mp4"

.\.venv-model\Scripts\python.exe .\xception_pilot\predict_video.py --video "$videoPath" --expected-label Fake
```

If either file is missing, report that error. A successful split-file download
does not prove that all dataset video files have been downloaded or validated.
The supplied expected label is only written to the report for comparison. It
is never used to choose the prediction or fed into the model.

Each run writes a unique UTC-timestamped directory below:

```text
C:\Projects\Deepfake_Detection_FYP\backend\xception_pilot\video_checks
```

It contains `video_report.json` and `face_alignment_preview.png`. The preview
retains small source-frame thumbnails and aligned face images for visual review.
Original videos are read only, not copied or changed. Individual crops are not
saved separately. This local debug retention is different from the upload API's
temporary-file cleanup policy. Outputs are not sent anywhere by this script.

To open the latest preview after a run:

```powershell
$latestRun = Get-ChildItem ".\xception_pilot\video_checks" -Directory |
    Sort-Object Name -Descending |
    Select-Object -First 1

Invoke-Item (Join-Path $latestRun.FullName "face_alignment_preview.png")
```

Inspect all accepted tiles: the lower image should show an upright, centered
face, without swapped left/right landmarks or excessive black padding. The
upper image shows the original frame with detection points. Check both the
console output and the JSON, including any skipped-frame reasons. A plausible
face preview alone does not establish classifier accuracy.

## Processing and result meaning

1. Read MP4 metadata and select up to 16 unique indices uniformly over the
   reported frame count. This is not a full-video decode or duplicate scan.
2. Decode each requested frame; reject failed or mismatched seeks.
3. Detect with YuNet at at most 640 pixels on the longest side, threshold 0.85.
   Preserve aspect ratio and map the five landmarks back to the original frame.
4. Require one detected face. Align from the original frame to 256x256 using a
   least-squares similarity transform. The target points and 1.3 expansion
   follow DeepfakeBench's release geometry; the warp uses linear interpolation
   and a zero border.
5. Convert BGR to RGB, scale to [0,1], normalize each channel with mean 0.5 and
   standard deviation 0.5, and arrange channels first.
6. If at least 8 crops are usable, run Xception on CPU in batches of one. Apply
   softmax to the two logits: class 0 Real, class 1 Fake. Average the frame Fake
   scores. Use 0.5 as an initial, untuned decision threshold.
7. If fewer than 8 crops are usable, report `insufficient_evidence`, with no
   prediction and no classifier inference. No-face is never interpreted as Real.

Initial engineering quality limits are: face box at least 32 pixels on each
side, eye separation at least 10 pixels, landmarks within the source frame,
alignment landmark RMS error at most 20 output pixels, and padding at most 25%.
The minimum 8 crops and these quality gates are **not validated scientific
cutoffs**. Record their version and assess rejection rates during validation.
Short videos use each sampled index only once; no frame duplication is used to
reach the minimum count.

Pilot resource limits: 200 MiB file, 120 seconds estimated duration, 50,000
reported frames, at most 3840x2160 pixels per decoded frame. These standalone
limits do not replace the existing API's separate size and duration limits.

`Fake score: 0.80` is a model score, not proof of manipulation, 80% measured
accuracy, or a calibrated 80% likelihood. A correct or incorrect result on two
examples is only a functional observation. Keep unsuccessful results too; do
not tune the threshold to make these two examples look correct.

Exit codes: 0 = scored; 2 = insufficient evidence (a report/preview is still
saved); 1 = failed input/model/runtime check. Timing separates model loading,
preprocessing and inference; it is not a reliable latency benchmark from one run.

## Differences from the pretrained model's reference pipeline

This is an **adapted deployment pipeline**, not an exact benchmark reproduction:

- The reference uses a dlib detector and selected landmarks 37, 44, 30, 49, 55
  from an 81-point predictor. YuNet's eye/nose/mouth landmarks have different
  semantics and positions. Matching destination geometry cannot remove that
  difference. YuNet can affect the checkpoint's accuracy; measure it.
- The reference selects the largest face. This pilot accepts only frames with
  exactly one detected face and adds quality checks.
- The reference Xception test configuration uses 32 frames; this pilot uses 16.
- No face identity tracking or temporal network is implemented. Scene changes
  may mix identities even when every frame has only one detected face.
- Audio is not analyzed.

If validation is weak, investigate these preprocessing differences and the
checkpoint's generalization before API integration. Fine-tuning later must use
the same recorded deployment preprocessing, or evaluate the impact of changing
it. Do not promise the original paper's benchmark results for this adapter.

## What happens after these two checks

Review alignment, frame rejection reasons, model scores, and CPU timing. Then
use a fixed, documented validation subset containing Real and all four FF++
manipulation methods. Report attempted/scored/rejected counts, per-video metrics
and timing. Tune only on validation. Once the pipeline is fixed, evaluate the
held-out FF++ test and Celeb-DF test sets, documenting any pretrained-checkpoint
selection history and limits on calling them fully independent.

Integrate the verified predictor into FastAPI only after reviewing these pilot
results. Run inference before temporary crop cleanup and update the Flutter
response handling. The current API can still correctly report
`model_loaded: false` because this standalone pilot has not integrated it.

## Attribution and references

The target alignment template and expansion are adapted from Zhiyuan Yan and
DeepfakeBench contributors' code, copyright (c) 2023 CUHK(SZ), CC BY-NC 4.0:

- [Reference preprocessing](https://github.com/SCLBD/DeepfakeBench/blob/6bc3ea01cdc768b17bd37b5e0eca85ecdfb10472/preprocessing/preprocess.py)
- [Reference image loader and normalization](https://github.com/SCLBD/DeepfakeBench/blob/6bc3ea01cdc768b17bd37b5e0eca85ecdfb10472/training/dataset/abstract_dataset.py)
- [Xception configuration](https://github.com/SCLBD/DeepfakeBench/blob/6bc3ea01cdc768b17bd37b5e0eca85ecdfb10472/training/config/detector/xception.yaml)
- [Upstream license](https://github.com/SCLBD/DeepfakeBench/blob/6bc3ea01cdc768b17bd37b5e0eca85ecdfb10472/LICENSE)

Changes dated 2026-11-05: replaced dlib landmarks with YuNet; independently
implemented the similarity solve with NumPy; added sampling, quality rejection,
CPU inference, score aggregation, previews, and JSON reports. The model and
architecture remain attributed research reuse, not original training work.
No endorsement by the upstream authors is implied. Retain the existing pilot's
full `LICENSE_DeepfakeBench.txt` and `ATTRIBUTION.md` with this add-on.

Other references:

- [OpenCV YuNet model and implementation](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- [OpenCV five-landmark output order](https://docs.opencv.org/4.x/df/d20/classcv_1_1FaceDetectorYN.html)
- [Official FF++ validation pairs](https://github.com/ondyari/FaceForensics/blob/master/dataset/splits/val.json)

## Verification of this add-on

See `VERIFICATION.txt` in the ZIP root for the completed local functional checks
and their limits. The user's FF++ and Celeb-DF videos are not available in the
test runtime, so dataset accuracy and Windows runtime behavior must be checked
on the laptop.
