# Emo-AEN Training Method

This document details a prescriptive, end-to-end method for training the Emo-AEN aesthetic ranking model. It complements `docs/training_guide.md` with concrete procedural steps and configuration tables.

## 1. Environment Provisioning
1. Create a dedicated Python 3.11 environment.
2. Install dependencies via `pip install -r requirements.txt`.
3. Verify GPU availability: `python -c "import torch; print(torch.cuda.is_available())"` should return `True`.
4. Set the project root as the working directory and initialize the `logs/` and `checkpoints/` folders.

## 2. Dataset Assembly
1. Gather photo pairs into the `training_files/` structure, where each folder contains `good/` and `bad/` subdirectories.
2. Ensure images are at least 224 px on the shortest side and encoded in sRGB.
3. Generate `splits.json` with explicit `{train, val, test}` assignments per folder.
4. Run the dataset audit script: `python tools/validate_dataset.py --root training_files --splits docs/splits.json`.

## 3. Preprocessing Pipeline
1. Resize all images to 256 px, then center-crop to 224×224.
2. Normalize channels using ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
3. Apply augmentations to the anchor (`good`) image only: random horizontal flip (p=0.5), small color jitter (±10 % brightness/contrast), and random rotation (±5°).
4. Persist preprocessing configuration in `configs/preprocess.yaml` for reproducibility.

## 4. Model Configuration
| Component | Setting |
|-----------|---------|
| Backbone | `mobilenet_v3_large` pretrained on ImageNet |
| Embedding dim | 512 |
| Head | Fully connected → L2 normalization |
| Loss | Hinge ranking loss with margin 0.2 |
| Optimizer | AdamW (lr=3e-4, weight_decay=1e-4) |
| Scheduler | Cosine annealing with warmup (5 epochs) |

Add an auxiliary regression head if mean opinion scores are available, weighting its MSE loss with λ=0.2 relative to the ranking loss.

## 5. Training Loop
1. Launch training: `python train.py --config configs/train.yaml`.
2. Use batch size 32 (16 pairs) with gradient accumulation if GPU memory is limited.
3. Clip gradients at a global norm of 5.0 to stabilize updates.
4. Evaluate on the validation split every epoch, logging hinge loss and pairwise accuracy.
5. Enable mixed precision (`torch.cuda.amp`) to reduce memory usage and speed up training.
6. Trigger early stopping when validation accuracy fails to improve by ≥0.5 % over five consecutive evaluations.

## 6. Checkpointing and Evaluation
1. Persist checkpoints every two epochs, keeping the best-performing model based on validation accuracy.
2. After training, run `python evaluate.py --checkpoint checkpoints/best.pt --split test`.
3. Record final metrics: pairwise accuracy, Kendall τ, and inference latency on desktop/mobile targets.
4. Document results in `docs/model_changelog.md` with dataset hash and configuration references.

## 7. Export and Deployment
1. Convert the final checkpoint to ONNX: `python export.py --checkpoint checkpoints/best.pt --onnx models/emo_aen.onnx --dynamic`.
2. Quantize the ONNX model with `python tools/quantize_onnx.py --input models/emo_aen.onnx --output models/emo_aen_int8.onnx`.
3. Validate parity between FP32 and INT8 exports using the evaluation script.
4. Bundle artifacts with metadata (version, training date, dataset hash) and update the web client manifest.

## 8. Automation Checklist
- [ ] Dataset validated and splits locked.
- [ ] Preprocessing config committed.
- [ ] Training logs archived in `weights & biases` project `emo-aen`.
- [ ] Best checkpoint and ONNX exports stored in `models/`.
- [ ] Documentation updated with metrics and configuration.

Following this method ensures reproducible Emo-AEN model training with explicit checkpoints for quality and deployment readiness.
