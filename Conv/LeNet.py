"""
复现LeNet-5卷积神经网络
用于MNIST数据集的经典神经网络
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1,6,5,1)
        self.s2 = nn.MaxPool2d(2)
        self.c3 = nn.Conv2d(6,16,5,1)
        self.s4 = nn.MaxPool2d(2)
        self.c5 = nn.Conv2d(16,120,5,1)
        self.flat = nn.Flatten()
        self.f6 = nn.Linear(120,84)
        self.fc = nn.Linear(84,10)
    def forward(self, input):
        o = self.c1(input)
        o = self.s2(o)
        o = self.c3(o)
        o = self.s4(o)
        o = self.c5(o)
        o = self.flat(o)
        o = self.f6(o)
        o = self.fc(o)
        return o

def train(net:Net, train_loader:DataLoader, test_loader:DataLoader, epoches:int):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr = 0.01)
    for i in range(epoches):
        net.train()
        for datas,targets in train_loader:
            pred = net(datas)
            f = loss(pred, targets)
            optimizer.zero_grad()
            f.backward()
            optimizer.step()
        test(net,test_loader,i)

def test(net:Net, test_loader:DataLoader, epoch_indx:int):
    total = 0
    correct = 0
    total_loss = 0.0
    net.eval()
    """
    使用CorssEntropyLoss模型最后一层不加softmax 输出logits分数即可(正常全连接层的输出一般就是未经归一化的logits) 该损失函数内部会做LogSoftmax
    CorssEntopyLoss对象一般是接收参数 preds (batch_size, num_classes) tragets (num_classes,) 不需要我们手动将preds从所属不同类别的概率向量 转换为标量
    """
    loss = nn.CrossEntropyLoss() 
    with torch.no_grad():
        for datas, targets in test_loader:
            total += targets.size(0) # batch_size，统计当前处理了的sample总数
            pred = net(datas)
            f = loss(pred,targets)
            total_loss += f.item()*datas.size(0) # Pytorch中损失函数默认对于batch平均，所以要乘回batch_size
            """
            沿着第1维取最大值 (batch_size,10)->(batch_size,1) 留下的是1是选中的索引
            _ 用于忽略返回的最大值 shape 也是 (batch_size,1)
            """
            _, indices = torch.max(pred, dim=1) 
            correct += torch.sum(indices == targets).item()
    acc = correct / total
    avg_loss = total_loss / total
    print(f"epoch:{epoch_indx},acc:{acc},avg_loss:{avg_loss}")


if __name__ == "__main__":
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(2) # 28*28->32*32
    ])

    net = Net()
    # 通过设定train的真/假，会自动使用官方划分的训练集和测试集
    TrainSet = torchvision.datasets.MNIST(root="./DataSets/MNIST",train=True,transform=trans)
    TestSet = torchvision.datasets.MNIST(root="./DataSets/MNIST",train=False,transform=trans)
    train_loader = DataLoader(dataset=TrainSet,shuffle=True,batch_size=64)
    test_loader = DataLoader(dataset=TestSet,shuffle=False,batch_size=64)

    train(net=net,train_loader=train_loader,test_loader=test_loader,epoches=200)
