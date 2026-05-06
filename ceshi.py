import torch
print(torch.__version__)
print(torch.version.cuda)         # 应显示 CUDA 版本
print(torch.backends.cudnn.enabled)  # 应返回 True
print(torch.cuda.is_available())
print("代码开始运行开头")