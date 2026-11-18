# Source attribution and changes

The Xception backbone is adapted from:

- Authors/project: Zhiyuan Yan and the DeepfakeBench contributors, SCLBD,
  The Chinese University of Hong Kong, Shenzhen.
- Copyright (c) 2023 CUHK(SZ).
- Source: https://github.com/SCLBD/DeepfakeBench/blob/6bc3ea01cdc768b17bd37b5e0eca85ecdfb10472/training/networks/xception.py
- Detector reference: https://github.com/SCLBD/DeepfakeBench/blob/6bc3ea01cdc768b17bd37b5e0eca85ecdfb10472/training/detectors/xception_detector.py
- Configuration reference: https://github.com/SCLBD/DeepfakeBench/blob/6bc3ea01cdc768b17bd37b5e0eca85ecdfb10472/training/config/detector/xception.yaml
- Original upstream link retained by DeepfakeBench:
  https://github.com/ondyari/FaceForensics/blob/master/classification/network/xception.py

DeepfakeBench's repository licence is CC BY-NC 4.0. The accompanying
`LICENSE_DeepfakeBench.txt` is copied from the same release and includes the
upstream terms and disclaimers. No author endorsement is implied.

Changes made for this DeepGuard loading pilot on 2026-11-05:

1. Removed only the benchmark-specific registry import and decorator from the
   backbone; layer definitions, parameter names and computations are retained.
2. Added a small inference wrapper preserving the `backbone.*` state keys.
3. Added a fixed-release downloader, recorded checksum, strict safe loading,
   synthetic CPU check and machine-readable local report.
4. Omitted the research framework's training loop and ImageNet initialization;
   all trained detector parameters and buffers must match before inference.

The original pretrained model and architecture are research reuse. The pilot
adapter and checks are integration work. Fine-tuning and video evaluation have
not been performed by this package.

Suggested references for the FYP documentation:

- Z. Yan, Y. Zhang, X. Yuan, S. Lyu, and B. Wu, "DeepfakeBench: A Comprehensive
  Benchmark of Deepfake Detection," NeurIPS, 2023.
- A. Rossler et al., "FaceForensics++: Learning to Detect Manipulated Facial
  Images," ICCV, 2019.
- F. Chollet, "Xception: Deep Learning with Depthwise Separable Convolutions,"
  CVPR, 2017.
