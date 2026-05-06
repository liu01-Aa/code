from .dsNet import DSNet
import torch
x = torch.rand(3,256,256)
x = DSNet(x)
print(x)