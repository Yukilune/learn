import torch
import torch.nn as nn
import torch.optim as optim

# 1. 定义 CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(6 * 14 * 14, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# 2. 准备练习数据：8 张图片，每张属于 10 个类别之一
torch.manual_seed(42)
images = torch.randn(8, 3, 28, 28)
labels = torch.randint(0, 10, (8,))

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# 3. 训练 5 轮
for epoch in range(5):
    model.train()

    # 前向传播：得到每张图片的 10 个类别分数
    outputs = model(images)

    # 计算损失
    loss = criterion(outputs, labels)

    # 反向传播和更新参数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 4. 统计预测正确的数量
    predictions = outputs.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    accuracy = correct / len(labels)

    print(
        f"第 {epoch + 1} 轮 | "
        f"损失：{loss.item():.4f} | "
        f"正确：{correct}/{len(labels)} | "
        f"准确率：{accuracy:.1%}"
    )