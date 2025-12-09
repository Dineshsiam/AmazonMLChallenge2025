
# AmazonMLChallenge2025

This repository contains the code, trained models, and datasets for the Amazon ML Challenge 2025.

## Contents

- `train.py` / `test_predictions.csv` — Training and evaluation scripts
- `model.py` / `best_fusion_model.pth` — Model definitions and trained model
- `dataset_loader1.py` — Dataset loader
- `preprocess.py` — Preprocessing scripts
- `train.csv`, `test.csv` — Datasets (tracked via Git LFS)

## Usage

1. Install dependencies (PyTorch, pandas, etc.)
2. Load data using `dataset_loader1.py`
3. Train model:  
   ```bash
   python train.py


4. Make predictions:

   ```bash
   python predict.py
   

## Notes

* Large files (models, CSVs) are tracked with Git LFS.
* Make sure Git LFS is installed before cloning the repo:

```bash
git lfs install
git lfs pull


