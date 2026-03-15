"""
线性层的demo
"""

import torch
import torchvision

dataset = torchvision.datasets.CIFAR10("./DataSets/CIFAR10",download=True,transform=torchvision.transforms.ToTensor(),train=True)
dataloader = torch.utils.data.DataLoader(dataset=dataset,batch_size=64,shuffle=True,drop_last=True)

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features=196608,out_features=1000)
    def forward(self,input):
        output = self.linear1(input)
        return output

m = MyModule()

for data in dataloader:
    imgs,labels = data
    imgs = torch.reshape(imgs,(1,1,1,-1))
    output = m(imgs)
    print(output.shape)