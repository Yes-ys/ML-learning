"""
演示神经网络的基本骨架:
torch.nn.Module的作用
"""
import torch.nn as nn
import torch

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        output = input+1
        return output

m = MyModule()
x = torch.tensor(1.0)
y = m(x)
print(y)