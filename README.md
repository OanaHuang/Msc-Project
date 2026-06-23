cat > README.md <<'EOF'
# MSc Project

This repository contains the initial code and experiment scripts for my MSc project on 2D human pose estimation using the MPII Human Pose Dataset.

## Project Overview

The current stage of the project focuses on building a minimal working pipeline for MPII-based human pose estimation. The repository includes scripts for:

- checking the processed MPII `.npz` file
- converting MPII annotations into a model-friendly format
- loading MPII data with PyTorch
- visualising MPII annotations
- training a simple baseline model
- visualising model predictions

## Repository Structure

```text
MSc-Project/
├── README.md
├── .gitignore
├── Literature/
│   └── paper.pdf
└── Datasets/
    └── MPII/
        └── Scripts/
            ├── check_npz.py
            ├── convert_mpii_to_npz.py
            ├── mpii_torch_dataset.py
            ├── test.py
            ├── train_baseline.py
            ├── visualize_mpii.py
            └── visualize_prediction.py
