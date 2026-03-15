"""
演示卷积神经网络的基本使用
"""
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 1, stride = 1)
    def forward(self, input):
        output = self.conv1(input)
        return output

Mydataset = torchvision.datasets.CIFAR10(root = "./DataSets/CIFAR10", train = True, transform = torchvision.transforms.ToTensor(), download = True)

dataloader = torch.utils.data.DataLoader(dataset = Mydataset, batch_size = 64, shuffle = True, drop_last = False)

writer = SummaryWriter("./logs/Conv")

m = MyModule()

step = 0

for data in dataloader:
    images, labels = data
    print("输入的shape:")
    print(images.shape)
    
    writer.add_images("Input", images, global_step = step)

    output = m(images)
    print("输出的shape:")
    print(output.shape)
    # writer.add_images("Output", output, global_step = step)

    output1 = torch.reshape(output,(-1, 3, 32, 32))
    writer.add_images("Output", output1, global_step = step)

    step += 1
