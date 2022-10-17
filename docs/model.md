# 模型

> 前向传播用于预测，反向传播用于训练。

## 定义模型

```python
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using { device } device")
```

定义模型，需要继承 `torch.nn.Module`，且需要实现以下两个函数：

* `__init__`：初始化网络模型中的各种层。
* `forward`：对输入数据进行相应的操作。

```python
class NeuralNetwork(nn.Module):
  def __init__(self):
    super(NeuralNetwork, self).__init__()
    self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(
      nn.Linear(in_features=28 * 28, out_features=512),
      nn.ReLU(),
      nn.Linear(in_features=512, out_features=512),
      nn.ReLU(),
      nn.Linear(in_features=512, out_features=10),
    )
  def forward(self, x):
    x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    return logits
```

```python
model = NeuralNetwork().to(device)
print(model)
```

我们可以将输入数据传入模型，会自动调用 `forward` 函数。模型会返回一个 `10` 维张量，其中包含每个类的原始预测值。我们使用 `nn.Softmax` 函数来预测类别的概率。

```python
X = torch.rand(1, 28, 28, device=device)
logits = model(X) # 调用forward函数
# 在第一个维度应用Softmax函数
pred_probab = nn.Softmax(dim=1)(logits)
# 最大概率值对应的下标
y_pred = pred_probab.argmax(1)
print(f"Predicted class: { y_pred }")
```

## 模型参数

使用 `parameters()` 或 `named_parameters()` 方法可以查看模型的参数。

```python
print(f"Model structure: { model }\n\n")
 
for name, param in model.named_parameters():
  print(f"Layer: { name } | Size: { param.size() } | Values : { param[:2] } \n")
```

## 自动微分

在训练神经网络时，最常用的算法是反向传播算法，模型参数会根据损失函数回传的梯度进行调整。为了计算这些梯度，PyTorch 有一个内置的微分引擎，称为 `torch.autograd`，它支持任何计算图的梯度自动计算。

下面定义了最简单的一层神经网络，具有输入 `x`、参数 `w` 和 `b` 以及一些损失函数。

```python
import torch
 
x = torch.ones(5)  # 输入
y = torch.zeros(3) # 期待的输出
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
```

在这个网络中，`w` 和 `b` 是我们需要优化的参数，设置了 `requires_grad=True` 属性。（可以在创建张量时设置该属性，也可以使用 `x.requires_grad_(True)` 来设置）

构建计算图的函数是 `Function` 类的一个对象。这个对象知道如何计算正向的函数，以及如何在反向传播步骤中计算导数，可以通过张量的 `grad_fn` 属性查看。

```python
print(f"Gradient function for z = { z.grad_fn }")
print(f"Gradient function for loss = { loss.grad_fn }")
```

### 计算梯度

为了优化神经网络中参数的权重，我们需要计算损失函数对参数的导数。我们可以调用 `loss.backward()` 来完成这一操作，在 `w.grad` 和 `b.grad` 中可以查看相应的导数值。

```python
loss.backward()
print(w.grad)
print(b.grad)
```

### 不使用梯度跟踪

默认情况下，所有张量的属性都设置为 `requires_grad=True`，用来跟踪它们的计算历史并支持梯度计算。但是，在某些情况下我们不需要这样做（如果这样做就会影响效率），例如，**模型训练完成后将其用于预测时**，只需要前向计算即可。具体操作如下：

```python
z = torch.matmul(x, w) + b
print(z.requires_grad)

with torch.no_grad():
  z = torch.matmul(x, w) + b
print(z.requires_grad)
```

或者使用 `detach()` 方法：

```python
z = torch.matmul(x, w) + b
z_det = z.detach()
print(z_det.requires_grad)
```

## 优化模型参数

训练模型是一个迭代过程；在每次迭代（epoch）中，模型对输出进行预测，首先计算猜测值与真实值的误差（损失），然后计算误差关于其参数的导数，最后使用梯度下降法优化这些参数。

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
 
training_data = datasets.FashionMNIST(
  root="data",
  train=True,
  download=True,
  transform=ToTensor()
)
 
test_data = datasets.FashionMNIST(
  root="data",
  train=False,
  download=True,
  transform=ToTensor()
)
 
train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)
class NeuralNetwork(nn.Module):
  def __init__(self):
    super(NeuralNetwork, self).__init__()
    self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(
      nn.Linear(28*28, 512),
      nn.ReLU(),
      nn.Linear(512, 512),
      nn.ReLU(),
      nn.Linear(512, 10),
    )
  def forward(self, x):
    x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    return logits
model = NeuralNetwork()
```

### 超参数

超参数是可调整的参数，不同的超参数值会影响模型训练和收敛速度。

这次训练，我们定义了以下超参数：

* 训练次数 `epochs`：迭代数据集的次数。
* 批处理大小 `batch_size`：每次传入网络中的样本数量。
* 学习率 `learning_rate`：在每个批次更新模型参数的程度。较小的值会产生较慢的学习速度，而较大的值可能会导致训练期间出现不可预测的行为。

```python
learning_rate = 1e-3
batch_size = 64
epochs = 5
```

### 优化循环

设置好超参数后，我们就可以使用优化循环来训练和优化我们的模型。

每个epoch包括以下两个循环：

训练循环：迭代训练数据集并尝试收敛到最佳参数。
验证/测试循环：迭代测试数据集以检查模型性能是否正在改善。

#### 损失函数

损失函数用来衡量模型预测得到的结果与真实值的差异程度，损失值越小越好。

常见的损失函数包括用于回归任务的 `nn.MSELoss`（均方误差）和用于分类的 `nn.NLLLoss`（负对数似然）。`nn.CrossEntropyLoss` 结合 `nn.LogSoftmax` 和 `nn.NLLLoss`。

这里我们将模型的输出 `logits` 传递给 `nn.CrossEntropyLoss`，进行归一化并计算预测误差。

```python
# 初始化损失函数
loss_fn = nn.CrossEntropyLoss()
```

#### 优化器

优化是在每个训练步骤中调整模型参数以减少模型误差的过程。在这里，我们使用 `SGD` 优化器；`torch.optim` 中提供了很多优化器，

例如 `ADAM` 和 `RMSProp`。

```python
# 传入需要优化的参数和学习率
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

#### 实践

在训练循环中，优化分三个步骤进行：

1. 调用 `optimizer.zero_grad()` 将模型参数的梯度归零。默认情况下梯度会累加。
2. 调用 `loss.backward()` 来反向传播预测损失。PyTorch 存储每个参数的损失梯度。
3. 计算梯度完成后，调用 `optimizer.step()` 来调整参数。

```python
# 优化模型参数
def train_loop(dataloader, model, loss_fn, optimizer, device):
  size = len(dataloader.dataset)
  for batch, (X, y) in enumerate(dataloader):
    X = X.to(device)
    y = y.to(device)
    # 前向传播，计算预测值
    pred = model(X)
    # 计算损失
    loss = loss_fn(pred, y)
    # 反向传播，优化参数
    optimizer.zero_grad() # 梯度归零
    loss.backward()       # 反向传播预测损失
    optimizer.step()      # 调整参数

    if batch % 100 == 0:
      loss, current = loss.item(), batch * len(X)
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# 测试模型性能
def test_loop(dataloader, model, loss_fn, device):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss, correct = 0, 0
  with torch.no_grad():
    for X, y in dataloader:
      X = X.to(device)
      y = y.to(device)
      # 前向传播，计算预测值
      pred = model(X)
      # 计算损失
      test_loss += loss_fn(pred, y).item()
      # 计算准确率
      correct += (pred.argmax(1) == y).type(torch.float).sum().item()

  test_loss /= num_batches
  correct /= size
  print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

我们初始化损失函数和优化器，并将其传递给 `train_loop` 和 `test_loop`。

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer, device)
    test_loop(test_dataloader, model, loss_fn, device)
print("Done!")
```

## 保存和加载模型

```python
import torch
import torchvision.models as models
```

PyTorch 模型将学习到的参数存储在内部状态字典中，称为 `state_dict`。

可以通过 `torch.save` 方法保存：`torch.save(model.state_dict(),model_path)`

保存和加载模型：
```python
torch.save(model, 'model.pth')
model = torch.load('model.pth')
```

加载模型分为两步：

1. 先加载模型中的 `state_dict` 参数，`state_dict=torch.load(model_path)`
2. 然后加载 `state_dict` 到定义好的模型中，`model.load_state_dict(state_dict,strict=True/False)`，`strict` 表示是否严格加载模型参数，`load_state_dict()` 会返回 `missing_keys` 和 `unexpected_keys` 两个参数

```python
# 样例代码如下
model = models.vgg16(pretrained=True) # pretrained=True加载预训练好的参数
torch.save(model.state_dict(), 'model_weights.pth')

# 要加载模型权重，首先需要创建一个相同模型的实例，然后使用load_state_dict()方法加载参数。
model = models.vgg16() # 不加载预训练好的参数
model.load_state_dict(torch.load('model_weights.pth'))
model.eval() # 将模型设置为测试模式，避免dropout和batch normalization对预测结果造成的影响
```

## 附完整代码

```python
import os
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
 
# 训练数据集
training_data = datasets.FashionMNIST(
  root="data",
  train=True,
  download=True,
  transform=ToTensor()  # 对样本数据进行处理，转换为张量数据
)
# 测试数据集
test_data = datasets.FashionMNIST(
  root="data",
  train=False,
  download=True,
  transform=ToTensor()  # 对样本数据进行处理，转换为张量数据
)
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
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     # 随机生成一个索引
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     # 获取样本及其对应的标签
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     # 设置标题
#     plt.title(labels_map[label])
#     # 不显示坐标轴
#     plt.axis("off")
#     # 显示灰度图
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()
 
# 训练数据加载器
train_dataloader = DataLoader(
  dataset=training_data,
  # 设置批量大小
  batch_size=64,
  # 打乱样本的顺序
  shuffle=True)
# 测试数据加载器
test_dataloader = DataLoader(
  dataset=test_data,
  batch_size=64,
  shuffle=True)
# 展示图片和标签
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")
 
# 模型定义
class NeuralNetwork(nn.Module):
  def __init__(self):
    super(NeuralNetwork, self).__init__()
    self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(
      nn.Linear(in_features=28 * 28, out_features=512),
      nn.ReLU(),
      nn.Linear(in_features=512, out_features=512),
      nn.ReLU(),
      nn.Linear(in_features=512, out_features=10),
    )

  def forward(self, x):
    x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    return logits

# 优化模型参数
def train_loop(dataloader, model, loss_fn, optimizer, device):
  size = len(dataloader.dataset)
  for batch, (X, y) in enumerate(dataloader):
    X = X.to(device)
    y = y.to(device)
    # 前向传播，计算预测值
    pred = model(X)
    # 计算损失
    loss = loss_fn(pred, y)
    # 反向传播，优化参数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch % 100 == 0:
      loss, current = loss.item(), batch * len(X)
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# 测试模型性能
def test_loop(dataloader, model, loss_fn, device):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss, correct = 0, 0

  with torch.no_grad():
    for X, y in dataloader:
      X = X.to(device)
      y = y.to(device)
      # 前向传播，计算预测值
      pred = model(X)
      # 计算损失
      test_loss += loss_fn(pred, y).item()
      # 计算准确率
      correct += (pred.argmax(1) == y).type(torch.float).sum().item()

  test_loss /= num_batches
  correct /= size
  print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == '__main__':
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Using {device} device")
  # 定义模型
  model = NeuralNetwork().to(device)
  # 设置超参数
  learning_rate = 1e-3
  batch_size = 64
  epochs = 5
  # 定义损失函数和优化器
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)
  # 训练模型
  for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer, device)
    test_loop(test_dataloader, model, loss_fn, device)
  print("Done!")
  # 保存模型
  torch.save(model.state_dict(), 'model_weights.pth')
```
