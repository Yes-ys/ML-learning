"""
用来生成mnist数据集的代码
"""

import torch
import torchvision

dataset = torchvision.datasets.MNIST(root="./DataSets/MNIST",download = True,transform=torchvision.transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset=dataset,batch_size=64)

datas,targets = next(iter(dataloader))
print(datas.shape)
print(targets.shape)