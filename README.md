DMSHA-Net: Dynamic Multi-Scale Hierarchical Attention Network

本项目主要用于植物叶片的精确分割任务。该模型通过集成动态多尺度特征融合与层次注意力机制，能有效应对复杂背景下的边缘提取挑战。

## 📊 支持的数据集
本项目已针对以下三个主流植物分割公开数据集进行了优化和测试：
*   **CVPPP**: 植物表型图像分析挑战赛数据集。
*   **KOMATSUNA**: 小松菜数据集，包含丰富的多叶片实例。
*   **MSU-PID**: 密歇根州立大学提供的植物图像数据集。

## 🛠️ 环境要求
*   **Python**: 3.x
*   **框架**: PyTorch
*   **主要依赖**: `torchvision`, `opencv-python`, `numpy`
*   *建议使用项目自带的虚拟环境配置。*

## 🚀 快速开始
1. **数据准备**: 将下载的数据集存放在项目根目录的 `Dataset/` 文件夹下。
2. **模型训练**: 针对 CVPPP 等数据集，直接运行以下脚本开始训练：
   ```bash
   python train_cvppp.py
