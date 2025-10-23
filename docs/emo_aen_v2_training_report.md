# Emo-AEN v2 Training Report (Attempt)

## Dataset Overview
- Source pairs located under `training_files/<pair_id>/good.jpg` and `bad.jpg`.
- Split configuration from `training_files/splits.json`:
  - Train: 80 pairs
  - Validation: 10 pairs
  - Test: 10 pairs
- No missing or corrupt pairs were detected while enumerating the dataset directories.

## Training Environment Blocker
- Planned configuration:
  - Backbone: MobileNetV3 (small variant)
  - Loss: Pairwise hinge (margin 0.2)
  - Optimizer: AdamW (lr=3e-4, weight_decay=1e-4)
  - Batch size: 16–32 with auto-selection
  - Training duration: up to 40 epochs with early stopping (patience 5)
  - Standard 224px preprocessing and light composition augmentations
- Attempted to install the required deep learning dependencies (`torch`, `torchvision`) via `pip` to begin training.
- The installation repeatedly failed because the environment is behind a proxy that returns **403 Forbidden** for package downloads. As a result, PyTorch cannot be installed and the training pipeline cannot be executed in this environment.

## Next Steps
- Obtain network access or a pre-provisioned environment with PyTorch (and CUDA if GPU acceleration is required).
- Once dependencies are available, rerun training with the configuration above, evaluate until validation pairwise accuracy ≥ 0.70, export to ONNX, quantize, and integrate the resulting model into the front end.
- After a successful run, update this report with actual training/validation metrics, export details, and latency measurements.
