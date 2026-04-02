"""
演示线性回归
"""

import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split # 用于按照比例随机提取数据集（可以用来在原始数据集上划分训练集和验证集）
import torch.optim as optim


class Net(nn.Module):
    def __init__(self, input_dim:int, output_dim:int):
        super().__init__() # 注意super()是函数，不是对象
        self.fc = nn.Linear(in_features=input_dim,out_features = output_dim,bias=True)
        # nn.Linear权重会使用输入特征来做均匀初始化，但是bias会全部初始化为0
        nn.init.uniform_(self.fc.bias, -0.1, 0.1) # 初始化偏置项，均匀分布在-0.1到0.1之间
    
    def forward(self, input):
        return self.fc(input)

class MyDataset(Dataset):
    def __init__(self, datas:torch.Tensor, labels:torch.Tensor):
        super().__init__()
        self.datas = datas
        self.labels = labels 

    def __getitem__(self, index:int):
        x, y = self.datas[index], self.labels[index]
        return x, y
    def __len__(self):
        return len(self.datas)

def generate_data(input_dim: int, output_dim: int, size: int): # 返回一个Dataset的对象以及一个“真实的”模型
    net = Net(input_dim = input_dim, output_dim = output_dim)
    x = torch.rand(size,input_dim) # 2维张量，size行、input_dim列 
    with torch.no_grad():
        y = net(x)
    dataset = MyDataset(datas=x,labels=y)
    train_dataset, val_dataset = random_split(dataset,[0.9,0.1]) # 接受int作为数据集的长度或者float作为划分比例，如果比例不足导致没有完全划分剩下的数据直接划分到最后一个子集
    return train_dataset, val_dataset, net # 此时两个数据集类型是Subset，这是Dataset的一个子类，random_split在原始数据集上生成新的数据集 这种方式会确保不会复制数据，提高性能

def train(mynet:Net, epoches:int, dataloader:DataLoader, testloader:DataLoader):
    """
        mynet.parameters() 能够自动找到当前网络的全部参数；
        是因为所有自定义的神经网络模型都继承自 torch.nn.Module 基类；
        而 nn.Module 内部实现了一套完整的参数注册与递归遍历机制；
        """
    best_mse = 1e8
    best_mae = 1e8

    optimizer = optim.SGD(mynet.parameters(),lr = 0.01) # 优化器和损失函数通常都定义在模型类的外部
    loss = nn.MSELoss() # 均方差损失
    for i in range(epoches):
        mynet.train() # 测试的时候使用了evl模式，要切换回来！
        for datas,labels in dataloader:
            o = mynet(datas)
            f = loss(input = o,target = labels)
            optimizer.zero_grad()
            f.backward()
            optimizer.step()

        mae,mse = test(mynet=mynet, rounds=i, dataloader=testloader)
        best_mae = min(best_mae,mae)
        best_mse = min(best_mse,mse)
    return best_mae,best_mse


def test(mynet:Net,rounds:int, dataloader:DataLoader):
    # 测试没必要使用epoches，epoches一般是指在整个数据集上的迭代次数
    targets = []
    preds = []

    mynet.eval()
    with torch.no_grad():
        for datas, labels in dataloader:
            o = mynet(datas) # 输出tensor's shape : (batch_size,shpae of a data)
            preds.append(o)
            targets.append(labels)
    
    t = torch.cat(tensors = targets, dim = 0) # 沿着第0维，batch_size维度拼接，得到的tensor's shape : (val_set_size, shape of a data)
    p = torch.cat(tensors = preds, dim = 0)
    mse = torch.mean((p-t)**2).item()
    mae = torch.mean(torch.abs(p-t)).item() # item是将tensor转换为普通的数值型，mean返回的是零维的tensor，只有一个元素
    print(f"rounds:{rounds},mse:{mse},mae:{mae}") # python特有的f-string
    return mae,mse

if __name__ ==  "__main__":
    train_dataset, val_dataset, true_net = generate_data(input_dim=50,output_dim=5,size=1000)
    train_dataloader = DataLoader(dataset=train_dataset,shuffle=True,batch_size=40)
    val_dataloader = DataLoader(dataset=val_dataset,shuffle=True,batch_size=40)
    net = Net(input_dim=50,output_dim=5)

    mae,mse =  train(mynet=net,epoches=100,dataloader=train_dataloader,testloader=val_dataloader)
    print(f"best_mae:{mae},best_mse:{mse}")