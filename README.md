# KalyaniMed

## An AI-Assisted Cervical Cytology Classification with Explainable AI (Grad-CAM)

## Overview
Automated cervical cancer screening using domain-specific transfer learning on microscopic histopathology/cytology images. This project implements a PyTorch classification pipeline paired with **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize cellular abnormalities and provide interpretable diagnostic support.

## Key Features
- **Transfer Learning:** Fine-tuned pre-trained CNN backbones (ResNet-50 / EfficientNet) for cervical cell classification.
- **Robust Evaluation:** Evaluated using Precision, Recall, F1-Score, and ROC-AUC metrics to minimize false negatives.
- **Explainable AI (XAI):** Integrated Grad-CAM to highlight nuclear enlargement and dyskeratotic regions influencing model predictions.

## Dataset
- **Primary Dataset:** PathMNIST / SIPAKMED Cytology Dataset
- **Classes:** Normal vs. Pathological / Dysplastic Cells

<img width="636" height="657" alt="sample1" src="https://github.com/user-attachments/assets/12900fac-4317-4f90-ab6c-743f3220e7c4" />

## Tech Stack
- **Language:** Python 3.10+
- **Frameworks:** PyTorch, torchvision
- **Computer Vision & Math:** OpenCV, NumPy, Matplotlib, scikit-learn

## Visualizations & Explainability
*(Insert your generated Grad-CAM heatmap here once ready)*
| Original Cell Image | Grad-CAM Heatmap Overlay |
| :---: | :---: |
| `![Original](./outputs/cell_sample.png)` | `![GradCAM](./outputs/gradcam_sample.png)` |

## Setup & Installation
```bash
git clone [https://github.com/your-username/cervical-cancer-histopathology-ai.git](https://github.com/your-username/cervical-cancer-histopathology-ai.git)
cd cervical-cancer-histopathology-ai
pip install -r requirements.txt
