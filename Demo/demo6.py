"""
演示dataloader的使用
"""
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_sets = torchvision.datasets.CIFAR10(root="./DataSets/CIFAR10", train = False, transform = torchvision.transforms.ToTensor(),download=False)
test_loader = DataLoader(dataset=test_sets, batch_size=4, shuffle=True, drop_last=False, num_workers=0)
writer = SummaryWriter("./logs/dataloader_logs")

step = 0
for data in test_loader:
    imgs, labels = data
    print(imgs.shape)
    print(labels)
    writer.add_images("test_images", imgs, global_step=step)
    step += 1
