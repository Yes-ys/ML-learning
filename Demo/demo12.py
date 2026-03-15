"""
一个实战的实例
CIFAR10分类网络的搭建
涉及Sequential容器以及tensorboard计算图的使用
"""

import torch.nn as nn
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Flatten(), # 注意这里的摊平处理
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )
    def forward(self,input):
        output = self.seq(input)
        return output

dataset = torchvision.datasets.CIFAR10(root = "./DataSets/CIFAR10",train = True,download=True,transform= torchvision.transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset=dataset,batch_size=64,shuffle=True)
writer = SummaryWriter("./logs/Model")


imgs, labels = next(iter(dataloader))
m = MyModule()

writer.add_graph(m, imgs) # 查看计算图，指定使用的模型和对应的输入

print(imgs.shape)
output = m(imgs)
print(output.shape)

writer.close()