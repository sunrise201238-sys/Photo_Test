# Emo-AEN Training Guide

## Dataset Structure
The Emo-AEN model uses **pairwise ranking** across curated pairs of "good" and "bad" compositions. Prepare the dataset directory as:

```
dataset/
  scene_0001/
    good.jpg
    bad.jpg
  scene_0002/
    good.png
    bad.png
  ...
```

Guidelines:
- Keep image resolutions at or above 224px on the shortest side (the pipeline resizes to 224×224).
- Ensure each pair shares the same scene/content; label quality differences only.
- Optional metadata JSON per pair can include tags (e.g. horizon tilt, crop type). Use these for stratified validation.
- Maintain a `splits.json` manifest that enumerates train/validation/test folders to make experiment reproduction easier.

### Validation Strategy
- Stratify the validation set across scene categories (portrait, landscape, architecture) so the ranking objective generalises.
- Preserve at least 10% of the scenes as a hold-out test set that is never touched during hyperparameter tuning.
- Track pairwise accuracy and hinge loss across both validation and test sets after every epoch.
- Store per-epoch metrics (CSV or JSON) to feed into dashboards and ensure early stopping triggers are transparent during audits.

## Preprocessing Pipeline
1. Convert images to sRGB and normalise to `[0, 1]` range.
2. Apply **composition-centric augmentations** to the bad image (small translations, 3–5° rotations, mild zoom-out) and complementary augmentations to the good version.
3. Resize/crop to `224 × 224` with `tf.image.resize_with_pad` or `torchvision.transforms.Resize`.
4. Standardise per-channel using ImageNet statistics if you initialise from MobileNet weights.

## Model Architecture
- **Backbone**: Lightweight MobileNetV3 or MobileNetV2 truncated at a mid-level feature block.
- **Dual-branch head**: Siamese-like structure processing the good/bad images with shared weights.
- **Pairwise ranking head**: Compute composition embeddings for each branch, then optimise a hinge loss with margin `m = 0.2` so `score(good) >= score(bad) + m`.
- **Auxiliary aesthetic head**: Shared trunk feeding an auxiliary regressor trained with MSE to predict an overall aesthetic score (if available).

## Training Hyperparameters
- Batch size: 16–32 pairs (32 pairs recommended when GPU memory allows).
- Learning rate: `3e-4` with cosine decay or plateau scheduler.
- Optimiser: AdamW (weight decay ≈ `1e-4`).
- Epochs: 20–40 with early stopping on validation pairwise accuracy (target ≥ 0.75).
- Data augmentations: random crop (±10%), horizontal flip, slight color jitter, perspective tweaks (<5%).

## Loss Functions
- **Ranking loss**: `L_rank = max(0, m - (score_good - score_bad))`.
- **Aux aesthetic loss**: `L_aes = ||aesthetic_good - target||^2 + ||aesthetic_bad - target_bad||^2`.
- **Total loss**: `L_total = L_rank + λ * L_aes` (λ ≈ 0.2).

## Evaluation Metrics
- Pairwise accuracy (percentage of pairs where `score_good > score_bad`).
- Kendall/Tau correlation between model ranking and human judgement (optional).
- Latency benchmarks on target hardware (goal: <300 ms desktop, <600 ms mobile for batch size 1).

## Exporting the Model
1. Trace or export the trained PyTorch model with `torch.onnx.export` (dynamic axes for batch).
2. Use ONNX Runtime quantisation (`quantize_dynamic` for FP16/INT8) or TensorFlow Lite converter if using TF.
3. Embed metadata:
   - Model name: `Emo-AEN`
   - Version: Semantic version string (e.g. `1.0.0`).
   - Training date, dataset snapshot hash.
4. Validate the exported model with ONNX Runtime Web or TF.js to confirm outputs match within tolerance.

## Packaging for the Web
- Place the quantised ONNX file at `models/` and update `modelVersion` in `main.js`.
- Provide a SHA256 checksum for release management.
- Optionally generate a smaller TF.js GraphModel for mobile browsers (swap `modelFormat` to `tfjs`).

## Automation Tips
- Use a notebook or script to automatically create augmented pairs and split train/val/test.
- Log validation accuracy per epoch; stop when improvement <0.5% over 5 epochs.
- Maintain a changelog referencing model version, dataset changes, and validation metrics.
