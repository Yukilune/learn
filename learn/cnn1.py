
import torch
import torch.nn as nn

# 1. 模拟输入：8 张 RGB 图片，每张 28×28
x = torch.randn(8, 3, 28, 28)

# 2. 卷积层：3 个输入通道，6 个输出通道
conv = nn.Conv2d(3, 6, kernel_size=3, padding=1)

# 3. 激活函数
relu = nn.ReLU()

# 4. 最大池化层
pool = nn.MaxPool2d(kernel_size=2, stride=2)

# 5. 依次执行并观察形状
y1 = conv(x)
print("卷积后：", y1.shape)

y2 = relu(y1)
print("ReLU 后：", y2.shape)

y3 = pool(y2)
print("池化后：", y3.shape)
