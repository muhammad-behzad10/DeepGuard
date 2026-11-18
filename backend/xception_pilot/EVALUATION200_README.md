# DeepGuard: broader FF++ validation (200 videos)

Prepared 9 November 2026. This add-on evaluates the existing pretrained Xception
pipeline. It does not train a model or change the app, SQLite history, model
weights, face policy, or threshold.

## Scope and selection

| Category | Videos |
| --- | ---: |
| Original / Real | 100 |
| Deepfakes | 25 |
| Face2Face | 25 |
| FaceSwap | 25 |
| NeuralTextures | 25 |
| Total | 200 |

All selected videos belong to the official FF++ **validation** split and use
c23 compression. This is a development study, not final test accuracy.

The exact previous 40-video plan is preserved as the `pilot40` cohort. The new
`additional160` cohort contains 40 additional source pairs, with 80 originals
and 80 manipulated videos (20 per method). There are 50 source pairs overall.
The additional selection and mixed execution order use fixed seed 20261109,
before any predictions. Related originals and manipulations are correlated;
200 videos do not represent 200 independent identities or scenes.

The study includes the earlier 40 cases: **do not report 240 distinct videos**.
Old predictions are not imported. All 200 cases are evaluated under this run's
recorded code, environment, and app engine; separate cohort metrics permit
comparison. No selected case is replaced because it is difficult or rejected.

## Laptop preparation

1. Save your source files. Wait for an active phone scan to finish.
2. Stop the FastAPI server with Ctrl+C in its own terminal. Stop a live Flutter
   run with `q` in the Flutter terminal. Neither is needed for this evaluation.
3. Connect the laptop charger and prevent sleep during a batch. You can unplug
   the phone. Run one evaluation process at a time and avoid other heavy CPU work.
4. Use the existing `.venv-model` interpreter. No new Python packages or model
   downloads are required for the already working integrated project.

## Install

Download `DeepGuard_Evaluation200.zip` into the Windows Downloads folder.
In a free VS Code PowerShell terminal:

```powershell
$evaluationDir = Join-Path "C:\Projects" ("DeepGuard_Evaluation200_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\DeepGuard_Evaluation200.zip" -DestinationPath $evaluationDir -ErrorAction Stop

cd "C:\Projects\Deepfake_Detection_FYP\backend"
& .\.venv-model\Scripts\python.exe "$evaluationDir\install_evaluation.py" --project-root "C:\Projects\Deepfake_Detection_FYP"
```

The installer adds only `evaluate_validation.py` and this README beside the
existing pilot scripts. Existing different files cause a stop before writes.
Reinstalling identical files is harmless. There is no backend or mobile patch.

## Prepare the plan first

```powershell
cd "C:\Projects\Deepfake_Detection_FYP\backend"

.\.venv-model\Scripts\python.exe .\xception_pilot\evaluate_validation.py --data-root "C:\Users\rajab\DeepGuard_Data" --prepare-only
```

Expected successful preflight:

```text
EVALUATION PREFLIGHT: PASSED (200 present, nonempty and hashed; videos not yet decoded)
100 Real + 100 Fake: Deepfakes 25 | Face2Face 25 | FaceSwap 25 | NeuralTextures 25
50 source pairs. Includes the original 40 cases plus 160 additions.
```

Preflight verifies the recorded official split fingerprints, source-ID separation,
selected files, and both model hashes. It records code/runtime fingerprints and
the plan. It does not prove that the videos decode correctly. Identical selected
bytes with conflicting Real/Fake labels stop the run; same-label byte duplicates
are retained and reported. This is not a complete dataset duplicate audit.

## Run 50 new cases at a time

After the preflight passes:

```powershell
.\.venv-model\Scripts\python.exe .\xception_pilot\evaluate_validation.py --data-root "C:\Users\rajab\DeepGuard_Data" --max-new-videos 50
```

Repeat that **same command** for each batch. It resumes at the next unfinished
case and normally takes four batches to finish 200 outcomes. An abstention or
error is an outcome too; it is not silently retried to improve the result.

To run all remaining cases in one session, omit `--max-new-videos 50`.
`Complete: False` is expected after an early batch. Partial metrics are marked
PROVISIONAL. `Complete: True` means every selected video has an outcome, not
that every video was scored or that the accuracy is satisfactory.

Observed earlier laptop processing times varied substantially. For planning,
allow roughly **2-5 hours total** for 200 cases, or **30-75 minutes per batch**.
These estimates are not guarantees. Model loading occurs once per invocation;
file hashing, CPU power/temperature and the number of usable faces affect time.

## Safe stopping and resuming

The easiest pause is to let a 50-case batch finish. You can also press Ctrl+C
once and wait for the PowerShell prompt. Completed per-video JSON records are
saved atomically. A video interrupted before its record commits runs again.
Rerun the same command to continue; avoid editing source, models, datasets or
dependencies between batches. A changed fingerprint causes a stop, not a merge
of incompatible results. A new intentional experiment needs a new `--run-dir`.
The OS lock prevents two evaluators writing the same run folder and releases on
exit. You do not need to delete a lock file after a crash.

## Inference and measurement

The evaluator calls the existing backend `XceptionEngine.analyse(video_path)`.
The expected dataset label is stored outside the engine and never passed to it.
Xception and YuNet are loaded once per batch, on CPU, with the engine's two
threads. Sampling remains 16 frames, at least 8 usable single-face frames,
aligned 256x256 crops, mean Fake score and a fixed 0.50 decision threshold.

This evaluates the same inference code used by the app, but **not** the HTTP,
upload, mobile UI or SQLite integration. The endpoint's 30-second/30-MB upload
restrictions are not applied. The existing engine limits still apply: 120
seconds, 200 MiB, 50,000 reported frames, and 3840x2160 total pixels per frame.
Clips exceeding them produce recorded errors rather than being substituted or
silently removed. Accepted clips are sampled across their full duration.
Network delay is not measured. The result `processing` time includes
face preprocessing and model inference; excludes uploads, API validation,
model loading, hashing, and report writing.

## Reports and interpretation

Default output folder:

```text
C:\Projects\Deepfake_Detection_FYP\backend\xception_pilot\validation_runs\ffpp_val200_seed20261109
```

- `subset_manifest.csv`: paths, labels, source pairs, cohorts, bytes and SHA-256.
- `run_config.json`: frozen plan, source/model hashes, runtime and settings.
- `reports/case_###.json`: engine output or error for each completed video.
- `validation_summary.json`: totals, metrics, coverage, cohort/method results,
  confusion matrix and processing-time median/p95.
- `video_results.csv`: one row per completed video for manual review.

Fake is the positive class. TN means Real predicted Real; FP means Real predicted
Fake; FN means Fake predicted Real; TP means Fake predicted Fake.
Accuracy, balanced accuracy, precision, recall, F1 and ROC-AUC use **scored
videos only**. Coverage is scored/attempted, reported overall and by class and
method. Insufficient-evidence clips are never relabelled Real. Errors remain
visible. Undefined metrics are null rather than invented zero values. Each
method has only 25 selected Fake videos; compare its recall and rejection rate,
not a single-class 'accuracy' as if it were a full binary evaluation.

This deliberately balanced sample has a different class prevalence from the
full dataset and deployment; precision should not be generalized to those.
YuNet differs from the pretrained reference dlib pipeline. Pretrained weights
and their original selection history prevent unsupported independent-training
or benchmark-reproduction claims. Scores remain uncalibrated.

After completion, share `validation_summary.json` and `video_results.csv`.
Investigate false positives, false negatives, method coverage and errors.
Preserve this baseline before any validation-based improvement. Keep the FF++
test videos and Celeb-DF external-test list separate for a later, explicitly
frozen final evaluation. Do not choose settings from those test results.

Exit code 0 means preparation or a batch completed without recorded errors; it
does not imply the full run completed. Code 3 reports video errors; 130 means
interruption; 1 means setup/preflight failed. Inspect the report counts.

## Sources

- Official FF++ split documentation: https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md
- Split files: https://github.com/ondyari/FaceForensics/tree/master/dataset/splits
- Model source: https://github.com/SCLBD/DeepfakeBench
- Test-set separation: https://scikit-learn.org/stable/common_pitfalls.html

Existing model/code licensing and attribution remain in the project. See the
package's VERIFICATION.md for exactly what was checked before delivery.
