"""
演示非线性激活函数的demo
"""

import torch.nn as nn
import torch

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
    def forward(self, input):
        output = self.relu(input)
        return output

input = torch.tensor([
    [-1, 1],
    [2, 2]
],dtype=torch.float32)

input = torch.reshape(input,(-1, 1, 2, 2))

m = MyModule()
output = m(input)
print(output)