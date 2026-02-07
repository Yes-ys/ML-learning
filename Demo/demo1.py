"""
用来演示read_data_insect.py，即读取昆虫数据集的demo
"""
from DataReader.read_data_insect import data_reader_insect 
from PIL import Image
import random

trian_dir = "DataSets/hymenoptera_data/train" # '/'能够被os正确处理，并且一般也建议这样使用
val_dir = "DataSets/hymenoptera_data/val" # 均采用相对路径

train_label_1 = "ants"
train_label_2 = "bees"

val_label_1 = "ants"
val_label_2 = "bees"

train_ants_data = data_reader_insect(trian_dir,train_label_1)
train_bees_data = data_reader_insect(trian_dir,train_label_2)
val_ants_data = data_reader_insect(val_dir,val_label_1)
val_bees_data = data_reader_insect(val_dir,val_label_2)

train_data = train_ants_data + train_bees_data
val_data = val_ants_data + val_bees_data

# 展示训练集或数据集中的随机一张图
train_idx = random.randint(0,train_data.__len__()-1)
val_idx = random.randint(0,val_data.__len__() - 1)

rand_train_img,train_label = train_data[train_idx] # 注意这里返回的是一个元组！！！
rand_val_img,val_label = val_data[val_idx]

rand_train_img.show()
rand_val_img.show()




