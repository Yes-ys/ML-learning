"""
用来演示如何使用tensorboard记录数据
主要涉及add_scalar和add_image的使用
"""
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")

img_pil1 = Image.open("/Users/yesys/Py_projects/MLpre/DataSets/hymenoptera_data/train/ants/7759525_1363d24e88.jpg")
img_pil2 = Image.open("/Users/yesys/Py_projects/MLpre/DataSets/hymenoptera_data/train/bees/150013791_969d9a968b.jpg")

img_np1 = np.array(img_pil1)
img_np2 = np.array(img_pil2)

writer.add_image("demo_img",img_np1,1,dataformats="HWC")
writer.add_image("demo_img",img_np2,2,dataformats="HWC")

for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)

writer.close()
