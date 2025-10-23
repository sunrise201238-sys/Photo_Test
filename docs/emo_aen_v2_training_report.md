# Emo-AEN v2 Training Report (Attempt)

## Dataset Overview
- Source pairs located under `training_files/<pair_id>/good.jpg` and `bad.jpg`.
- Split configuration from `training_files/splits.json`:
  - Train: 80 pairs
  - Validation: 10 pairs
  - Test: 10 pairs
- No missing or corrupt pairs were detected while enumerating the dataset directories.

## Updated Training Process – Google Colab Workflow
Given the persistent proxy restrictions in the local environment, future Emo-AEN v2 training runs should be executed in Google Colab. The steps below describe the revised process end to end.

### 1. Launching the Colab Runtime
1. Navigate to [https://colab.research.google.com](https://colab.research.google.com) and start a new notebook.
2. In the menu, choose **Runtime → Change runtime type**, set **Hardware accelerator** to **GPU**, and ensure the Python version is 3.10 or newer.

### 2. Preparing the Dataset
1. From this repository, create a ZIP archive of `training_files/` (including `splits.json`).
2. Upload the archive to Google Drive, then mount Drive inside Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Extract the dataset into `/content/data/training_files` and verify that each pair directory still contains `good.jpg` and `bad.jpg`.

### 3. Environment Setup in Colab
Run the following cell to install the required dependencies directly within the Colab runtime:
```python
!pip install --upgrade pip setuptools wheel
!pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
!pip install -r https://raw.githubusercontent.com/<your-org>/Photo_Test/main/requirements.txt
```
If `requirements.txt` is not publicly accessible, download it locally from this branch and upload it to Colab before installation.

### 4. Executing the Training Script
1. Clone this repository (or upload the project folder) into `/content/workspace`.
2. Run the training entry point with the prescribed hyperparameters:
   ```python
   %cd /content/workspace/Photo_Test
   !python tools/train_emo_aen.py \
       --dataset-root /content/data/training_files \
       --splits-file /content/data/training_files/splits.json \
       --backbone mobilenet_v3_small \
       --margin 0.2 \
       --lr 3e-4 \
       --weight-decay 1e-4 \
       --min-batch 16 \
       --max-batch 32 \
       --max-epochs 40 \
       --patience 5
   ```
3. Monitor validation pairwise accuracy each epoch; stop early when accuracy ≥ 0.70.

### 5. Exporting and Quantizing in Colab
After training produces the best checkpoint (e.g., `checkpoints/emo_aen_v2_best.pt`), export and quantize inside Colab:
```python
!python tools/export_to_onnx.py \
    --checkpoint checkpoints/emo_aen_v2_best.pt \
    --output models/emo_aen_v2.onnx \
    --dynamic-batch

!python tools/quantize_onnx.py \
    --input models/emo_aen_v2.onnx \
    --output models/emo_aen_v2_int8.onnx \
    --dtype int8
```
Record inference latency in the notebook by timing a batch of forward passes with both the FP32 and quantized models.

### 6. Retrieving Artifacts
1. Copy the final ONNX file (`emo_aen_v2.onnx` or `emo_aen_v2_int8.onnx`), logs, and metrics back to Google Drive.
2. Download the artifacts to your workstation and commit them under `models/` in this repository.
3. Update the frontend configuration (`assets/js/main.js` and service worker cache key, if applicable) and append the collected metrics to this report once local validation is complete.

## Pending Metrics
Training has not yet been executed in Colab. Populate the sections below after completing the first successful Colab run.

- **Best validation accuracy:** _TBD_
- **Best validation hinge loss:** _TBD_
- **Test accuracy:** _TBD_
- **Export date:** _TBD_
- **Quantization format:** _TBD_
- **Latency (FP32 / quantized):** _TBD_
