# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] - 2026-04-15
### Added
- SVTR backbone with tiny / small / base variants
- Feature Rearrangement Module (FRM)
- Semantic Guidance Module (SGM) with auxiliary CTC loss
- FocalCTCLoss with label smoothing and focal weighting
- CutMix width-axis augmentation
- Beam search decoder with lexicon regex filter and ctcdecode fast path
- 5x TTA at inference (brightness/contrast sweep + majority vote)
- ONNX FP32 and INT8 dynamic quantization export
- CLAHE → bilateral → deskew → unsharp preprocessing pipeline
- Full pytest suite covering codec, loss, model forward, preprocess, config
- GitHub Actions CI with Python 3.10/3.11 matrix and Codecov integration
