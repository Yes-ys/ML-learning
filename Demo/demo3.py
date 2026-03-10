"""
演示trasnform的使用
"""

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor

pil_img = Image.open("../DataSets/hymenoptera_data/train/bees/17209602_fe5a5a746f.jpg")

tensor_trans = ToTensor()

tensor_img = tensor_trans(pil_img)

writer = SummaryWriter("logs")
writer.add_image("tensor_img", tensor_img)
writer.close()
