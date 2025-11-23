# Salient Object Detection (SOD) Project  

## Overview  
Hi! This project implements a Salient Object Detection (SOD) model using deep learning. The goal is to identify and outline the most visually important object/s within an image by producing a binary saliency mask. The project includes multiple model architectures, training scripts, evaluation pipelines, and a demonstration notebook.

The implementation follows project requirements by developing the model from scratch using PyTorch, with no pretrained encoders. The dataset used is ECSSD, exported locally from Deep Lake.

## Project Structure  
- `config.py`  
  Centralized paths and hyper-parameters.
- `data_loader.py`  
  Loads images/masks from local dataset, applies resizing and normalization.
- `train.py`  
  Handles training loops, resuming checkpoints, logging, and saving best model.
- `sod_model.py`  
  Contains multiple model variants:
  - `SODNetBaseline`: basic U-Net style encoder-decoder
  - `SODNetImproved`: adds BatchNorm + Dropout to reduce overfitting
  - `SODNetImprovedV2`: adds residual connections + dropout  
- `evaluate.py`  
  Computes IoU, Precision, Recall, F1, MAE and saves visual results.


## Dataset
The project uses the ECSSD dataset. Images and masks are saved in the following structure:

```
dataset_ecssd/
    images/
    masks/
```

All scripts expect this dataset to be prepared before running training.

## Models & Experiments
Three trained models are included:

| Model Name | Architecture | Notes |
|------------|--------------|-------|
| `best_model.pth` | SODNetBaseline | Basic model, starting point |
| `sod_improved_dropout03_full.pth` | SODNetImproved | Best-performing model |
| `sod_improved_v2.pth` | SODNetImprovedV2 | Residual blocks + dropout |

The best model achieved significantly higher IoU and F1 performance than baseline.

## Training
To train from scratch:

```
python train.py
```

Training will automatically:
- Log metrics to a CSV file
- Save the best-performing checkpoint
- Resume training if a checkpoint already exists

## Evaluation
To evaluate model performance:

```
python evaluate.py
```

This generates:
- Overall performance metrics (IoU, Precision, Recall, etc.)
- Output visualizations
- Saliency mask overlays

## Demo / Notebook Use
A Google Colab notebook is included for live demonstration. It allows users to:
- Load a trained model
- Upload their own image
- Display predictions and overlays
- Compare models side-by-side

This was designed for presentation purposes and requires mounting Google Drive.

## Requirements
Core dependencies include:

```
torch
torchvision
numpy
Pillow
matplotlib
tqdm
segmentation-models-pytorch
```

Install all requirements using:

```
pip install -r requirements.txt
```

## Purpose
This project was assigned as part of the 'XponentL Data' & 'Genpact' Xponian Internship to the AI Engineering Stream. The objective was to design a working CNN model from scratch, improve performance through multiple revisions, and document results.

Special thanks to my mentor Fatmir Nuredini for guiding me through the project!

Copyright &copy; 2025 Tringa Desku