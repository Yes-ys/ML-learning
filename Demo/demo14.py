"""
用于演示pytorch自动微分的demo
"""
import torch

x = torch.tensor([1,2,3,4],dtype=torch.float32)
x.requires_grad_(True)
print(x.grad)

y = 2 * torch.dot(x, x)
y.backward()
print(x.grad) # tensor([ 4.,  8., 12., 16.])
print(x.grad == 4 * x)

# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_() # 使用该方法之后得到的梯度是tensor([1., 1., 1., 1.])，否则tensor([ 5.,  9., 13., 17.])
y = x.sum()
y.backward()
print(x.grad)