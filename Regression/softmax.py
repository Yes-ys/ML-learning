"""
一个softmax回归的demo
"""

import torchvision
import torch
import torch.nn as nn
import torch.optim as optim

train_sets = torchvision.datasets.FashionMNIST(root="./DataSets/FashionMNIST",download=True,train=True,transform=torchvision.transforms.ToTensor())
test_sets = torchvision.datasets.FashionMNIST(root="./DataSets/FashionMNIST",download=True,train=False,transform=torchvision.transforms.ToTensor())


class softmax(nn.Module):
    def __init__(self):
        # 模型里可以定义的基本属性
        super().__init__()
        self.linear = nn.Linear(in_features=28*28,out_features=10) # (size:(64,28*28)->(64,10)) bath_size的维度会保持
        self.softmax = nn.Softmax() # 类而非方法
        self.flat = nn.Flatten() # 类而非方法
        self.optimizer = optim.SGD(self.parameters(), lr=0.01) # 使用优化器
        self.train_loader = torch.utils.data.DataLoader(dataset=train_sets,batch_size=64,shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_sets,batch_size=64,shuffle=True)
    def forward(self,input):
        output = self.flat(input) # 摊平特征变量（28*28->784, size:(64,784)） pytorch中大多数模型都是默认批量处理，第一维度是batch_size
        output = self.linear(output)
        output = self.softmax(output)
        return output
    def train(self):
        f = nn.CrossEntropyLoss()
        for data in self.train_loader:
            imgs,labels = data
            # print(imgs.shape)
            y = self.forward(imgs)
            loss = f(input=y, target=labels)

            self.optimizer.zero_grad() # ！使用优化器
            loss.backward()
            self.optimizer.step()
            # print(y.shape)
    def test(self):
        total = 0.0
        correct = 0.0
        for data in self.test_loader:
            imgs,labels = data
            y = self.forward(imgs)
            _,index = torch.max(dim=1,input=y) # 测试的处理方法
            total += labels.size(0)
            correct += (index == labels).sum().item()
        acc = correct/total
        print("准确率：")
        print(acc)

m = softmax()

for epoch in range(100):
    m.train()
    m.test()