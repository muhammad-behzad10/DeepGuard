# DeepGuard

DeepGuard is an Android research prototype for assessing facial manipulation in videos. A Flutter app sends a local MP4 to a FastAPI service running on a laptop. The service samples face frames and runs a pretrained Xception classifier on the CPU. Results can be saved in the phone's local SQLite history.

The app and backend are integrated. Model fine-tuning has not been performed. A prediction is an experimental model assessment, not proof that a video is authentic or manipulated.

## Features

- Select and preview a local MP4 video.
- Check backend availability and model readiness.
- Connect through USB forwarding or a shared Wi-Fi network, with saved connection settings.
- Sample 16 frames, detect faces with YuNet and align accepted faces to 256 x 256 pixels.
- Run pretrained Xception inference when at least 8 usable face frames are available.
- Display a predicted label, raw Fake score, usable frame count and processing time.
- Return an insufficient-evidence result when there are too few usable faces.
- Save and reopen scan result details using SQLite on the device.
- Handle invalid videos, unavailable services and failed requests.

The backend removes temporary uploads and keeps face crops in memory during analysis. Local history stores result details rather than copies of the uploaded videos. It is device-local, without account synchronisation.

## How analysis works

The backend validates an upload and selects uniformly spaced frame indices. YuNet detects faces and supplies landmarks for alignment. The current policy accepts frames with one detected face and applies additional quality checks. Accepted faces are converted to RGB and normalised for Xception.

The classifier produces two outputs: class 0 is Real and class 1 is Fake. The video score is the mean Fake softmax score across accepted frames. The initial decision threshold is 0.50 and has not been tuned. The score is not a calibrated probability or an accuracy percentage. This is sampled-frame facial analysis; it does not analyse audio or every frame in the video.

## Code locations

| Location | Purpose |
| --- | --- |
| `mobile/lib/main.dart` | Flutter application entry point |
| `mobile/lib/screens/` | App screens, including connection settings and history |
| `mobile/lib/services/` | Network, persistence and upload services |
| `mobile/test/` | Flutter tests and fixtures |
| `backend/main.py` | FastAPI entry point and upload endpoint |
| `backend/api_contract.py` | API response models |
| `backend/inference_service.py` | Model lifecycle and video inference integration |
| `backend/video_validation.py` | Basic video validation |
| `backend/xception_pilot/xception_backbone.py` | Adapted upstream Xception architecture |
| `backend/xception_pilot/xception_loader.py` | Checkpoint verification and strict weight loading |
| `backend/xception_pilot/video_preprocessing.py` | Frame sampling, YuNet and face alignment |
| `backend/xception_pilot/evaluate_validation.py` | Resumable validation evaluation |
| `backend/models/` | Required local model files; model binaries are not committed |

## Requirements and model files

The project has been run on Windows with Python 3.11, a CPU PyTorch environment, Flutter and a physical Android phone. The tested laptop has 8 GB RAM. No dedicated GPU is required for this inference demo. Processing time depends on the CPU workload and video; this is not real-time detection.

This repository contains source code, not a complete environment backup. Datasets, Python virtual environments, model binaries, APKs, local databases and generated previews/reports are excluded from Git.

The backend requires these exact files in `backend/models/`:

- `xception_best.pth`: pretrained DeepfakeBench v1.0.1 Xception checkpoint. Source: [official release asset](https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/xception_best.pth).
- `face_detection_yunet_2026may.onnx`: the YuNet checkpoint used by this project. Restore the already verified file from the project backup. A fresh checkout also needs this external file; it is not supplied by Git cloning. Its expected SHA-256 is recorded as `YUNET_SHA256` in `video_preprocessing.py`.

The loaders check expected fingerprints. Do not rename a different model file to bypass those checks. The Xception fingerprint recorded by this project is a file-identity check, not a publisher signature.

## Run the backend

Use the project root as the starting directory. For a new Windows checkout, create the environment and install the recorded model environment dependencies:

```powershell
cd backend
py -3.11 -m venv .venv-model
.\.venv-model\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements_model_snapshot.txt
.\.venv-model\Scripts\python.exe -m pip check
```

The snapshot records the developer's environment. Recreating it on a different machine still requires package availability and compatible system tools. If `.venv-model` already works, reuse it rather than reinstalling packages for a demonstration.

With dependencies installed, the supplied checker can download and verify the Xception checkpoint if it is missing:

```powershell
.\.venv-model\Scripts\python.exe .\xception_pilot\check_xception.py --download
```

That command tests loading with a synthetic tensor; it does not measure accuracy or supply YuNet. Place the verified YuNet file as described above before starting the service.

```powershell
.\.venv-model\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Run one server process and leave its terminal open. Check [health](http://127.0.0.1:8000/health) for `model_loaded: true`. Interactive API documentation is at [the local docs page](http://127.0.0.1:8000/docs). The video endpoint is `POST /upload-video`. The API limit is 30,000,000 bytes per MP4.

## Run the Android app

In a separate PowerShell terminal, from the project root:

```powershell
cd mobile
flutter pub get
flutter analyze
flutter test
```

For a development run, connect an Android phone with USB debugging enabled:

```powershell
$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
& $adb devices
$deviceId = Read-Host "Enter the device ID shown above"
& $adb -s $deviceId reverse tcp:8000 tcp:8000
flutter run -d $deviceId
```

In the app, open Connection Settings, select USB, test the service and save. USB mode uses `http://127.0.0.1:8000` through ADB reverse. Reapply forwarding after reconnecting the phone when needed.

For Wi-Fi, put the phone and laptop on the same network. Use `ipconfig` to find the laptop's Wi-Fi IPv4 address. Select Wi-Fi in Connection Settings, enter that address, then Test and Save. The server must stay running and Windows Firewall must allow the local connection. These settings support a local demonstration, not an internet-facing production deployment.

A release APK can be built with `flutter build apk --release`. Open an already installed release app from its phone icon; `flutter run` is not needed to use that installed app. The laptop backend is still required for new scans.

## Recorded verification and validation

Recorded Flutter verification completed with no analyzer issues and 47 passing automated tests. Physical-phone checks covered USB and Wi-Fi scans, saved settings, persistent scan history, backend unavailability and recovery. These are recorded checks on the developer's setup, not a claim of continuous integration or compatibility with every device.

The broader evaluation used a selected FaceForensics++ c23 **validation** subset: 100 Real videos and 100 Fake videos, with 25 Fake examples from each of Deepfakes, Face2Face, FaceSwap and NeuralTextures.

| Measure | Recorded result |
| --- | --- |
| Attempted | 200 videos |
| Scored | 184 videos |
| Insufficient evidence | 16 videos |
| Processing errors | 0 |
| Prediction coverage | 92.00% |
| Accuracy on scored videos | 91.30% |
| Balanced accuracy on scored videos | 91.30% |
| False positive rate on scored videos | 15.22% |
| Fake precision on scored videos | 86.54% |
| Fake recall on scored videos | 97.83% |
| Fake F1 on scored videos | 0.9184 |
| ROC-AUC on scored videos | 0.9850 |

These results are from `ffpp_val200_seed20261109`. Insufficient-evidence videos are excluded from the classification metrics and are accounted for separately through coverage. The subset includes the earlier 40-video validation pilot; these are not independent studies. This is not an independent final test or a measure of general real-world accuracy. Raw local reports are excluded from the initial source upload.

## Limitations and planned work

- Real videos can be classified as Fake and manipulated videos can be missed.
- Multiple faces, small faces, compression and pose changes can reduce usable evidence.
- YuNet alignment differs from the upstream benchmark's dlib preprocessing; this integration needs its own evaluation.
- Sixteen sampled frames can miss a manipulation appearing briefly between samples.
- CPU inference and local network transfer add latency; the backend must remain running.
- Local history has no account sync. Login is outside the current project scope.
- Fine-tuning in Google Colab, independent final evaluation and broader external-dataset evaluation remain planned work.

## Sources and acknowledgements

- [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench): Xception architecture and pretrained v1.0.1 checkpoint. The adapted architecture records upstream commit `6bc3ea01cdc768b17bd37b5e0eca85ecdfb10472` and author credit in its header. See `backend/xception_pilot/ATTRIBUTION.md` and `LICENSE_DeepfakeBench.txt` for the upstream notices.
- [FaceForensics](https://github.com/ondyari/FaceForensics): dataset, official split lists and earlier Xception implementation referenced by the upstream code.
- [OpenCV Zoo](https://github.com/opencv/opencv_zoo): YuNet face detection. Keep the YuNet licence supplied with the model.
- [AppDeepfakeDetection](https://github.com/AyeshaMalikAyesha/AppDeepfakeDetection): Flutter reference project used with the author's permission.

DeepGuard integrates these components into its video-analysis application. It does not claim that the project author invented Xception, trained the supplied checkpoint or authored all third-party code. Original source notices and licences remain attached to their components.
