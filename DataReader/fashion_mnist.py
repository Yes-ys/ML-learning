"""
用来下载fashion_mnist数据集的脚本
"""

import torchvision
import torch

train_sets = torchvision.datasets.FashionMNIST("./DataSets/FashionMNIST", train=True, download=True, transform=torchvision.transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(train_sets, batch_size=64, shuffle=True)

for i in range(1):
    images, labels = next(iter(dataloader))
    print(images.shape, labels.shape)
    print(images)
    print(labels)