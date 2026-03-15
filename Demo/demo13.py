"""
损失函数与反向传播的demo
"""

import torch.nn as nn
import torch
import torchvision

input = torch.tensor([1,2,3],dtype=torch.float32)
target = torch.tensor([3,4,5],dtype=torch.float32)

input = torch.reshape(input, (-1, 1, 1, 3))
target = torch.reshape(target, (-1, 1, 1, 3))

f = nn.L1Loss() # 注意loss function本身也是一个对象
loss = f(input, target) # 同样使用__call__的方式实现=
print(loss)
print(target.shape)

# 关于交叉熵
input = torch.tensor([0.1,0.2,0.3])
target = torch.tensor([1])

input = torch.reshape(input,(1,3))
f = nn.CrossEntropyLoss()
loss = f(input,target)
print(loss)

# 结合demo12的CIFAR10分类网络（交叉熵）

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
    
dataset = torchvision.datasets.CIFAR10(root="./DataSets/CIFAR10", train=True, download=True,transform=torchvision.transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset=dataset,batch_size = 1)
m = MyModule()

for data in dataloader:
    imgs, labels = data
    input = m(imgs)
    input = torch.reshape(input,(1,10))
    loss = f(input,labels)
    print(loss)