# DMSHA-Net: Dynamic Multi-Scale Hierarchical Attention Network

This is the official implementation of **DMSHA-Net**, specifically designed for precision plant leaf segmentation tasks. By integrating **Dynamic Multi-scale Feature Fusion** and **Hierarchical Attention** mechanisms, the model effectively addresses the challenges of edge extraction in complex backgrounds.

## 📊 Supported Datasets
The project has been optimized and evaluated on three mainstream open-source plant segmentation datasets:
*   **CVPPP**: The Plant Phenotyping Evaluation (CVPPP) dataset.
*   **KOMATSUNA**: A dataset for komatsuna (Japanese mustard spinach) segmentation with various leaf instances.
*   **MSU-PID**: Michigan State University Plant Image Dataset.

## 🛠️ Environment Requirements
*   **Python**: 3.x
*   **Framework**: PyTorch
*   **Main Dependencies**: `torchvision`, `opencv-python`, `numpy`
*   *Note: It is recommended to use the provided virtual environment (`myenv`) for configuration.*

## 🚀 Quick Start
1. **Data Preparation**: Place the downloaded datasets into the `Dataset/` folder in the project root directory.
2. **Model Training**: For the CVPPP dataset, run the following script to start training:
   ```bash
   python train_cvppp.py
