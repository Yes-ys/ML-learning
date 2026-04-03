import torchvision
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=7,stride=2,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(), # 沿着0维展平，线性层接收的是一维参数
            nn.Linear(in_features=256*3*3,out_features=1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024,out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512,out_features=100)
        )
    def forward(self, input):
        return self.seq(input)

def generate_data():
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_dataset = torchvision.datasets.CIFAR100(root="./DataSets/CIFAR100",download=True,train=True,transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR100(root="./DataSets/CIFAR100",download=True,train=False,transform=transform_test)
    pin_memory = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=64,shuffle=True,pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=64,shuffle=False,pin_memory=pin_memory)
    return train_loader,test_loader

def train(net:Net,train_loader:DataLoader,test_loader:DataLoader,epoch:int,device:torch.device):
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(),lr=0.005,momentum=0.9,weight_decay=5e-4)
    crossloss = nn.CrossEntropyLoss()
    for i in range(epoch):
        net.train()
        running_loss = 0.0
        total_train = 0
        for datas,targets in train_loader:
            datas = datas.to(device)
            targets = targets.to(device)
            preds = net(datas)
            loss = crossloss(preds,targets)

            if not torch.isfinite(loss):
                raise RuntimeError(f"训练出现非有限损失: {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item() * datas.size(0)
            total_train += datas.size(0)

        avg_train_loss = running_loss / total_train
        test(net,test_loader,i,device,avg_train_loss)

def test(net:Net,test_loader:DataLoader,epoch:int,device:torch.device,avg_train_loss:float):
    total = 0
    correct = 0
    total_loss = 0.0
    crossloss = nn.CrossEntropyLoss()
    net.eval()
    with torch.no_grad():
        for datas,targets in test_loader:
            datas = datas.to(device)
            targets = targets.to(device)
            total += datas.size(0)
            preds = net(datas)
            loss = crossloss(preds,targets) # loss是0维tensor
            total_loss += (loss.item())*datas.size(0)

            _,indices = torch.max(preds,dim=1)
            correct += torch.sum(targets==indices).item()
    acc = correct/total
    avg_loss = total_loss/total
    print(f"训练轮次:{epoch},训练损失:{avg_train_loss:.4f},测试正确率:{acc:.4f},测试平均损失:{avg_loss:.4f}")
            
if __name__ == "__main__":
    device = get_device()
    print(f"当前设备: {device}")
    net = Net()
    train_loader,test_loader = generate_data()
    train(net,train_loader,test_loader,epoch=200,device=device)