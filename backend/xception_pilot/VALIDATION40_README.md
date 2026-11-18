# DeepGuard: fixed 40-video validation pilot

This add-on evaluates the existing Xception video pipeline on a small,
reproducible FF++ validation subset. It does not train the model, change
preprocessing, tune the threshold, or integrate the FastAPI/Flutter app.

## Why this is the next step

The two laptop reports supplied on 2026-11-05 are internally consistent:

| Observation | Real `720.mp4` | Fake `720_672.mp4` |
| --- | --- | --- |
| Mean Fake score | 0.526975868 | 0.998139672 |
| Prediction at 0.5 | Fake (false positive) | Fake (true positive) |
| Usable faces | 16/16 | 16/16 |
| Frame-score range | 0.0424-0.9719 | 0.9871-1.0000 |
| Average outside-image padding | 11.60% | 10.60% |
| Maximum outside-image padding | 17.25% | 17.52% |
| Mean landmark alignment RMS | 4.01 pixels | 4.23 pixels |

The checkpoints, preprocessing settings, input shapes, normalization metadata,
and aggregation agree between the reports. Recomputing the averages reproduces
the reported video scores. All recorded crops passed the current engineering
quality gates. That is not proof of correct reference preprocessing or accuracy.
The similar padding levels do not establish padding as the cause of the Real
video error. Only the Real preview was visually supplied and inspected; a Fake
preview has not been visually verified.

Keep these two outcomes as development observations, including the incorrect
one. Do not present them as an accuracy study, and do not change threshold or
aggregation merely to correct this pair.

## Install

Extract the ZIP and copy the two files from its `xception_pilot` folder into:

```text
C:\Projects\Deepfake_Detection_FYP\backend\xception_pilot
```

- `validate_subset.py`
- `VALIDATION40_README.md`

They must sit beside the existing `predict_video.py`, `video_preprocessing.py`,
`xception_loader.py`, and `xception_backbone.py`. This package imports those
working modules. Keep their source attribution and licenses.

Use the existing `.venv-model` Python environment. No additional packages,
model downloads, mobile connection, Flutter process, or FastAPI server are
needed for this standalone evaluation. The command uses the explicit Python
path even if the terminal prompt says `(.venv)`.

## Step 1: prepare and check the fixed subset

In VS Code PowerShell:

```powershell
cd "C:\Projects\Deepfake_Detection_FYP\backend"

.\.venv-model\Scripts\python.exe .\xception_pilot\validate_subset.py --data-root "C:\Users\rajab\DeepGuard_Data" --prepare-only
```

It verifies the official split pairs, checks source-ID separation, creates the
fixed list, and checks the selected files and model fingerprints. Expected:

```text
SUBSET PREFLIGHT: PASSED (40 present, nonempty, hashed; videos not yet decoded)
20 Real + 20 Fake: Deepfakes 5 | Face2Face 5 | FaceSwap 5 | NeuralTextures 5
```

Preparation checks only these selected files, not the whole dataset. It does
not decode the videos or prove that the dataset is free of duplicate content.
If a selected file is missing, empty, or a split/model differs, report the
error. The script does not silently substitute an easier/different video.

## Step 2: run validation

After preflight passes, run:

```powershell
.\.venv-model\Scripts\python.exe .\xception_pilot\validate_subset.py --data-root "C:\Users\rajab\DeepGuard_Data"
```

The same preflight runs again and guards the saved plan. Xception and YuNet
load once, and videos run sequentially with two CPU threads and batch size one.
Each completed video prints one summary line. Keep the laptop connected to
power and prevent it sleeping during the run. Avoid running another validation
instance or heavy CPU work simultaneously if you want comparable timing.

The two observed laptop preprocessing-plus-inference times were about 26 and
33 seconds per video. Forty similar videos would take roughly 18-22 minutes
of processing; allow about 20-35 minutes in practice. This is a planning estimate,
not a measured total: other videos, CPU temperature, and system load can change it.

## Sampling policy

The script reads the existing official `train.json`, `val.json`, and `test.json`
from `FaceForensics++\splits`. Semantic fingerprints preserve the actual ID
pair membership while allowing whitespace, list order and pair direction changes.
The script never opens training or test videos.

Using fixed seed `20261105`, it samples 10 pairs from the official validation
pairs, before calculating any predictions. Both original IDs in each pair are
included: 20 Real videos. It then includes one manipulated version per original
ID, with methods assigned reproducibly so there are exactly five of each method:

| Category | Videos |
| --- | ---: |
| Original / Real | 20 |
| Deepfakes | 5 |
| Face2Face | 5 |
| FaceSwap | 5 |
| NeuralTextures | 5 |
| Total | 40 |

The dataset is balanced at the video-label level. These are only 10 source
pairs, not 40 independent people or source scenes. Do not treat correlated
frames or related manipulations as independent statistical observations.
No Celeb-DF videos are used at this stage.

## Results and interrupted runs

Default output directory:

```text
C:\Projects\Deepfake_Detection_FYP\backend\xception_pilot\validation_runs\ffpp_val40_seed20261105
```

Main files:

- `subset_manifest.csv`: fixed paths, labels, source pairs, file sizes and SHA-256.
- `run_config.json`: plan, source-code hashes, model hashes, versions and settings.
- `reports\case_###.json`: each completed video's scores or rejection/error record.
- `validation_summary.json`: counts, metrics, coverage, methods and timing.
- `video_results.csv`: one row per completed video, suitable for review.

Reports are saved after each video. To stop, press Ctrl+C and wait for the
terminal prompt. To resume, run the same validation command. Completed records
are reused and an interrupted in-progress video is rerun. A crash may leave a
temporary `.tmp` file; committed per-video JSON records remain the resume source.

Run one instance at a time. The script verifies the selected file contents,
code, package versions, thread settings and model hashes before reusing results.
If any differ, it stops. Preserve the original run and use a new explicit
`--run-dir` for a changed experiment; do not overwrite the old evidence.
Completed rejected/error cases are reused as outcomes too. Fixing data or code
requires a new run directory so results are not silently mixed.

No previews are generated in bulk, and source videos are never modified or
copied. For a specific interesting result, use the existing `predict_video.py`
command to produce a preview for that video.

## Reading the metrics

Each scored video contributes one observation. The model still samples up to
16 unique frames, requires at least eight usable faces, averages frame softmax
Fake scores, and uses threshold 0.50. Individual frames do not count as separate
videos in the reported accuracy.

- **Coverage:** scored videos divided by attempted videos; inspect Real and Fake
  coverage separately. Pending videos are not included in attempted counts.
- **False positive rate:** scored Real videos incorrectly labeled Fake.
- **Fake recall:** scored Fake videos correctly labeled Fake.
- **Balanced accuracy:** average of Real specificity and Fake recall.
- **Precision and F1:** use Fake as the positive class.
- **ROC-AUC:** ranking of scored Real versus Fake videos; ties count as half a win.
- **Per-method results:** correct Fake, missed-as-Real, rejected/error, and recall
  for each manipulation method. Five examples per method is a small pilot.
- **p50/p95 processing time:** preprocessing plus inference on scored clips;
  excludes model loading, preflight hashing, writing reports, and network/API/UI.

Accuracy and the other classifier metrics use scored videos only. Rejected
videos are not relabeled Real and errors are not silently discarded: both are
counted beside coverage. A high scored-only accuracy with poor coverage is not
a successful overall result. Undefined metrics are JSON `null` / console
`undefined` (for example, ROC-AUC when only one class has scored videos).

The `complete` flag becomes true when all 40 cases have a terminal outcome,
including errors or insufficient evidence. It does not mean all 40 were scored
or that detection quality passed a target. Exit 0 means processing completed
without runtime errors (some videos may still be rejected); exit 3 means one or
more per-video errors; exit 1 means setup/preflight failed; exit 130 means an
interrupted run. Read the counts, not just the exit code.

## What to share after the run

Send `validation_summary.json` and `video_results.csv`, plus any terminal error.
Review false positives, misses by method, rejection rates and CPU timing before
deciding on preprocessing changes, threshold calibration, or FastAPI integration.

These are development validation results. They are not final test accuracy,
general reliability on internet videos, or a reproduction of published scores.
The pretrained checkpoint's selection history and the known dlib-versus-YuNet
preprocessing difference remain relevant. Final evaluation should use a larger,
documented protocol with its limitations, after choices are fixed.

## Sources and verification

Official split source:
https://github.com/ondyari/FaceForensics/tree/master/dataset/splits

The existing DeepGuard pilot retains the DeepfakeBench model and preprocessing
attribution. This add-on supplies subset selection, evaluation, reporting and
resume handling; it does not claim original training of the pretrained model.

See `VERIFICATION.txt` at the ZIP root for completed local software checks.
The user's dataset is on their laptop and was not evaluated in this runtime.
