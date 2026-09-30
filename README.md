# KalyaniMed

## An AI-Assisted Histopathology Classification with Explainable AI (Grad-CAM)

## Overview
Automated cervical cancer screening using domain-specific transfer learning on microscopic histopathology images. This project implements a PyTorch classification pipeline paired with **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize cellular abnormalities and provide interpretable diagnostic support.

## Key Features
- **Transfer Learning:** Fine-tuned pre-trained CNN backbones (ResNet-18) for cervical cell classification.
- **Robust Evaluation:** Evaluated using Precision, Recall, F1-Score, and ROC-AUC metrics to minimize false negatives.
- **Explainable AI (XAI):** Integrated Grad-CAM to highlight nuclear enlargement and dyskeratotic regions influencing model predictions.

## Dataset
- **Primary Dataset:** PathMNIST Dataset
- **Classes:** Normal vs. Pathological / Dysplastic Cells

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

⚠️ Disclaimer

KalyaniMed is an educational and research-oriented machine learning project. Its predictions are not intended to replace professional medical diagnosis or clinical decision-making. The model has not been clinically validated and should not be used for real-world diagnosis.