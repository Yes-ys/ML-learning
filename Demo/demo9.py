"""
池化层神经网络的demo
"""
import torch.nn as nn
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=3,ceil_mode=True)
    def forward(self, input):
        output = self.pool1(input)
        return output

test_input = torch.tensor([
    [1, 2, 0, 3, 1],
    [0, 1, 2, 3, 1],
    [1, 2, 1, 0, 0],
    [5, 2, 3, 1, 1],
    [2, 1, 0, 1, 1]
], dtype=torch.float32)

test_input = torch.reshape(test_input, (-1, 1, 5, 5)) # 输入神经网络的标准tensor形式，N C H W

m = MyModule()
output = m(test_input)
print(output)

# 显示池化层就像1080p->720p那样压缩视频的效果

writer = SummaryWriter("./logs/Pool")
dataset = torchvision.datasets.CIFAR10(root="./DataSets/CIFAR10",train = True,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=64, shuffle=True)
mm = MyModule()


step = 0

# 注意add_image只支持单张图片通常是C H W，add_images才能多张拼在一起
for data in dataloader:
    imgs, labels = data
    writer.add_images("before_pool", imgs, global_step=step)
    output_imgs = mm(imgs)
    writer.add_images("after_pool", output_imgs, global_step=step)
    step+=1
