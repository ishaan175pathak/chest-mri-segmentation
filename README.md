# Chest MRI Segmentation with a 3D U-Net

A PyTorch + [MONAI](https://monai.io/) pipeline for volumetric segmentation of chest MRI scans. The notebook covers the full workflow: data loading from JSON splits, 3D preprocessing and augmentation, a residual 3D U-Net, Dice + cross-entropy training, best-checkpoint selection, and slice-level visualization of predictions.

> **Status: baseline / work in progress.** The pipeline runs end to end, but the logged model performance is low (see [Results](#results)). This repo is best read as a clean, reproducible starting point for 3D medical segmentation rather than a finished model.

## Highlights

- **3D residual U-Net** (MONAI) with 5 resolution levels: `16 → 32 → 64 → 128 → 256` channels, 2 residual units per level
- **Combined Dice + Cross-Entropy loss** (`DiceCELoss`) for class-imbalanced foreground/background segmentation
- **MONAI transform pipeline**: intensity scaling, volume resizing to `128×128×64`, random flips and 90° rotations
- **Best-model checkpointing** on validation Dice, with training-loss and validation-Dice curves
- **Qualitative inspection**: MRI / ground truth / prediction / overlay panels for a mid-volume slice

## Pipeline

| Stage | Details |
|---|---|
| Data | Train / val / test file lists loaded from `data/splits/{train,val,test}.json` (each entry has an `image` and `label` path) |
| Preprocessing | `LoadImaged` → `EnsureChannelFirstd` → intensity clip and scale `[0, 3000] → [0, 1]` → `Resized` to `(128, 128, 64)` |
| Augmentation (train only) | `RandFlipd` (p=0.5), `RandRotate90d` (p=0.5) |
| Model | MONAI `UNet`, `spatial_dims=3`, 1 input channel, 1 output channel, strides `(2, 2, 2, 2)` |
| Loss / optimizer | `DiceCELoss(sigmoid=True)`, Adam, learning rate `1e-4` |
| Metric | `DiceMetric` (background excluded), sigmoid + 0.5 threshold on predictions |
| Training | Up to 50 epochs (configured), DataLoader batch size 4 |

## Results

Logged over the first 10 epochs of training:

| Epoch | Training loss | Validation Dice |
|---|---|---|
| 1 | 0.949 | 0.055 |
| 5 | 0.924 | 0.061 |
| 10 | 0.917 | 0.061 |

Training loss decreases steadily, but validation Dice plateaus around 0.06, so the model is not yet segmenting reliably. Likely areas to investigate: label/foreground sparsity after resizing, learning rate and loss weighting, and patch-based training instead of whole-volume resizing.

## Repository structure

```
chest-mri-segmentation/
├── MRI_Chest_Segmentation.ipynb   # full training + evaluation notebook
├── scr/data/                      # data directory
├── requirements.txt
└── README.md
```

The dataset is **not** included. To run the notebook, create `data/splits/train.json`, `val.json` and `test.json`, each a list like:

```json
[{"image": "path/to/scan.nii.gz", "label": "path/to/mask.nii.gz"}]
```

## Getting started

```bash
git clone https://github.com/ishaan175pathak/chest-mri-segmentation.git
cd chest-mri-segmentation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook MRI_Chest_Segmentation.ipynb
```

A CUDA GPU is strongly recommended for 3D training. The notebook falls back to CPU automatically.

## Roadmap

- [ ] Diagnose the low Dice plateau (label sparsity, loss weighting, learning-rate schedule)
- [ ] Switch to patch-based training with sliding-window inference
- [ ] Report Dice on the held-out test split
- [ ] Add a reproducible data-preparation script and dataset documentation

## Tech stack

Python · PyTorch · MONAI · NumPy · Matplotlib · Jupyter

## Author

**Ishaan Pathak** · [GitHub](https://github.com/ishaan175pathak)
