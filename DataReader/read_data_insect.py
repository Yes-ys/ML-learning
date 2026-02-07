from torch.utils.data import Dataset
from PIL import Image
import os

class data_reader_insect(Dataset):
    def __init__(self, root_dir, label_dir): # 初始化的时候将数据的地址组织成一个列表
        self.root_dir = root_dir
        self.dir_list = os.listdir(os.path.join(root_dir,label_dir))
        self.label_dir = label_dir
    def __getitem__(self,idx): # 使用idx访问列表，得到图片地址，读取到图片
        img_dir = self.dir_list[idx]
        img_dir = os.path.join(self.root_dir,self.label_dir,img_dir) # os.listdir只返回文件名，需要补全路径！！！
        img = Image.open(img_dir)
        label = self.label_dir
        return img,label # 最后返回数据和标签
    def __len__(self):
        return len(self.dir_list)