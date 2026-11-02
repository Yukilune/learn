
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 定义 CNN 模型
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


# 2. 创建模型和模拟数据
torch.manual_seed(42)

model = SimpleCNN()
images = torch.randn(8, 3, 28, 28)
labels = torch.randint(0, 10, (8,))

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. 训练10轮
for epoch in range(10):
    model.train()

    # 前向传播：得到原始类别分数
    outputs = model(images)

    # 计算损失：这里不要提前加 Softmax
    loss = criterion(outputs, labels)

    # 反向传播与参数更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"第 {epoch + 1} 轮，损失：{loss.item():.4f}")

# 5. 查看第一张图片的预测分数和概率
model.eval()
with torch.no_grad():
    outputs = model(images)
    probabilities = torch.softmax(outputs, dim=1)

    print("\n第一张图片的原始分数:")
    print(outputs[0])

    print("\n第一张图片的类别概率:")
    print(probabilities[0])

    predicted_class = probabilities[0].argmax().item()
    print("\n预测类别:", predicted_class)
    print("概率总和：", probabilities[0].sum().item())
