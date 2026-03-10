"""
演示transforms和数据集一起使用，即dataset的使用
"""
import torchvision

train_sets = torchvision.datasets.CIFAR10("./DataSets/CIFAR10", train=True, download=True)
test_sets = torchvision.datasets.CIFAR10("./DataSets/CIFAR10", train=False, download=True)
# 注意，同一个数据集的root应该是一样的，否则会触发数据集的再次下载，应该只用train作为标志来区分训练集和测试集

img, label = train_sets[0]

print(img)
print(label)
print(train_sets.classes[label])

img.show()
