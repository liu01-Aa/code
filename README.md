This project is primarily focused on precision plant leaf segmentation tasks. The model integrates **Dynamic Multi-scale Feature Fusion** and **Hierarchical Attention** mechanisms to effectively address the challenges of edge extraction in complex backgrounds.

## 📊 Supported Datasets
The project has been optimized and evaluated on three mainstream open-source plant segmentation datasets:
*   **CVPPP**: The Plant Phenotyping Evaluation dataset.
*   **KOMATSUNA**: A dataset for komatsuna (Japanese mustard spinach) segmentation, containing various leaf instances.
*   **MSU-PID**: A plant image dataset.

## 🛠️ Environment Requirements
*   **Python**: 3.8.0
*   **Framework**: PyTorch 1.7.0
*   **Main Dependencies**: `torchvision`, `opencv-python`, `numpy`
*   *Note: It is recommended to use the provided virtual environment configuration.*

## 🚀 Quick Start
1. **Data Preparation**: Place the downloaded datasets into the `Dataset/` folder in the project root directory.
2. **Model Training**: For datasets such as CVPPP, run the following script directly to start training:
   ```bash
   python train_cvppp.py
