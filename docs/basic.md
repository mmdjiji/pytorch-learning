# 基础

## 线性代数

### 张量（Tensor）

* 0维张量（标量）
* 1维张量（向量）
* 2维张量（矩阵）
* 3维张量（时间序列）
* 4维张量（图像）
* 5维张量（视频）

```python
import torch
import numpy as np
```

```python
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(f"Tensor from Data:\n { x_data } \n")
```

```python
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"Tensor from Numpy:\n { x_np } \n")
```

```python
# 保留原有张量的形状和数据类型，填充1
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n { x_ones } \n")
```

```python
# 显式更改张量的数据类型，随机填充
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n { x_rand } \n")
```

```python
# 创建2行3列的张量
shape = (2, 3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n { rand_tensor } \n")
print(f"Ones Tensor: \n { ones_tensor } \n")
print(f"Zeros Tensor: \n { zeros_tensor } \n")
```

```python
# 将张量移动到GPU上
if torch.cuda.is_available():
  tensor = tensor.to("cuda")
  print('success moving to gpu')
else:
  print('failed moving to gpu, use cpu only')
```

```python
tensor = torch.ones(4, 4)
print(f"First row: { tensor[0] }")
print(f"First column: { tensor[:, 0] }")
print(f"Last column: { tensor[..., -1] }")
tensor[:,1] = 0
print(tensor)
```

```python
# 在第1个维度拼接，即水平方向
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
```

```python
# 矩阵相乘，y1、y2和y3的值相同
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
 
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)
print(y1)
```

```python
# 矩阵逐元素相乘，z1、z2和z3的值相同
z1 = tensor * tensor
z2 = tensor.mul(tensor)
 
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(z1)
```

```python
# 只有一个值的张量，可以通过item属性转换为数值
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))
```

在CPU上的张量和NumPy数组共享它们的内存位置，改变一个会改变另一个。

```python
t = torch.ones(5)
print(f"t: { t }")
n = t.numpy()
print(f"n: { n }")
```

```python
t.add_(1)
print(f"t: { t }")
print(f"n: { n }")
```

```python
n = np.ones(5)
print(f"n: { n }")
t = torch.from_numpy(n)
print(f"t: { t }")
```

```python
np.add(n, 2, out=n)
print(f"t: { t }")
print(f"n: { n }")
```

## 数值计算

* https://github.com/Theano/Theano
* https://github.com/aesara-devs/aesara

## 数据集

### 加载数据集

```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
 
# 训练数据集
training_data = datasets.FashionMNIST(
  root="data", # 数据集下载路径
  train=True, # True为训练集，False为测试集
  download=True, # 是否要下载
  transform=ToTensor() # 对样本数据进行处理，转换为张量数据
)
# 测试数据集
test_data = datasets.FashionMNIST(
  root="data",
  train=False,
  download=True,
  transform=ToTensor() 
)
```

可视化：
```python
# 标签字典，一个key键对应一个label
labels_map = {
  0: "T-Shirt",
  1: "Trouser",
  2: "Pullover",
  3: "Dress",
  4: "Coat",
  5: "Sandal",
  6: "Shirt",
  7: "Sneaker",
  8: "Bag",
  9: "Ankle Boot",
}
# 设置画布大小
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
  # 随机生成一个索引
  sample_idx = torch.randint(len(training_data), size=(1,)).item()
  # 获取样本及其对应的标签
  img, label = training_data[sample_idx]
  # 添加子图
  figure.add_subplot(rows, cols, i)
  # 设置标题
  plt.title(labels_map[label])
  # 不显示坐标轴
  plt.axis("off")
  # 显示灰度图
  plt.imshow(img.squeeze(), cmap="gray")
plt.show()
```

### 自定义数据集

自定义数据集，需要继承 `Dataset` 类，并实现三个函数：

* `__init__`：实例化Dataset对象时运行，完成初始化工作。
* `__len__`：返回数据集的大小。
* `__getitem__`：根据索引返回一个样本（数据和标签）。

```python
import os
import pandas as pd
from torchvision.io import read_image
 
class CustomImageDataset(Dataset):
  def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    # 读取标签文件
    self.img_labels = pd.read_csv(annotations_file)
    # 读取图片存储路径
    self.img_dir = img_dir
    # 数据处理方法
    self.transform = transform
    # 标签处理方法
    self.target_transform = target_transform

  def __len__(self):
    return len(self.img_labels)

  def __getitem__(self, idx):
    # 单张图片路径
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    # 读取图片
    image = read_image(img_path)
    # 获得对应的标签
    label = self.img_labels.iloc[idx, 1]
    if self.transform:
      image = self.transform(image)
    if self.target_transform:
      label = self.target_transform(label)
    # 返回一个元组
    return image, label
```

## 数据加载器

### torch.utils.data.DataLoader

根据数据集生成一个可迭代的对象，用于模型训练。

常用参数：

* dataset (Dataset) ：定义好的数据集。
* batch_size (int, optional)：每次放入网络训练的批次大小，默认为1.
* shuffle (bool, optional) ：是否打乱数据的顺序，默认为False。一般训练集设置为True，测试集设置为False。
* num_workers (int, optional) ：线程数，默认为0。在Windows下设置大于0的数可能会报错。
* drop_last (bool, optional) ：是否丢弃最后一个批次的数据，默认为False。

两个工具包，可配合DataLoader使用：

* enumerate(iterable, start=0)：输入是一个可迭代的对象和下标索引开始值；返回可迭代对象的下标索引和数据本身。
* tqdm(iterable)：进度条可视化工具包

```python

from torch.utils.data import DataLoader
 
data_loader = DataLoader(
  dataset=MyDataset,
  batch_size=16,
  shuffle=True,
  num_workers=0,
  drop_last=False,
)
```

### 加载数据

在训练模型时，我们通常希望以小批量的形式传递样本，这样可以减少模型的过拟合。

```python
from torch.utils.data import DataLoader
 
train_dataloader = DataLoader(
  dataset=training_data, 
  # 设置批量大小
  batch_size=64, 
  # 打乱样本的顺序
  shuffle=True)
test_dataloader = DataLoader(
  dataset=test_data, 
  batch_size=64,
  shuffle=True)
```

### 遍历 DataLoader

将数据加载到DataLoader后，每次迭代一批样本数据和标签（这里批量大小为64），且样本顺序是被打乱的。

```python
# 展示图片和标签
train_features, train_labels = next(iter(train_dataloader))
# (B,N,H,W)
print(f"Feature batch shape: { train_features.size() }")
print(f"Labels batch shape: { train_labels.size() }")
# 获取第一张图片，去除第一个批量维度
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: { label }")
```