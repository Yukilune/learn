
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. 设置设备：优先使用 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前使用设备：", device)

# 2. 加载 MNIST 训练集和测试集
transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# 每批处理 64 张图片
train_loader = DataLoader(
    train_data, batch_size=64, shuffle=True, num_workers=0
)

test_loader = DataLoader(
    test_data, batch_size=64, shuffle=False, num_workers=0
)

# 3. 定义 CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)    # [B, 1, 28, 28] -> [B, 16, 28, 28]
        x = self.relu(x)
        x = self.pool(x)     # [B, 16, 28, 28] -> [B, 16, 14, 14]

        x = self.conv2(x)    # [B, 16, 14, 14] -> [B, 32, 14, 14]
        x = self.relu(x)
        x = self.pool(x)     # [B, 32, 14, 14] -> [B, 32, 7, 7]

        x = self.flatten(x)  # [B, 32, 7, 7] -> [B, 1568]
        x = self.fc(x)       # [B, 1568] -> [B, 10]

        return x


model = SimpleCNN().to(device)

# 4. 设置损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 训练模型
epochs = 3

for epoch in range(epochs):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计本批次的损失和准确率
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    print(
        f"第 {epoch + 1}/{epochs} 轮 | "
        f"训练损失：{avg_loss:.4f} | "
        f"训练准确率：{accuracy:.2%}"
    )

# 6. 在测试集上评估模型
model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        test_correct += (predictions == labels).sum().item()
        test_total += labels.size(0)

test_accuracy = test_correct / test_total

print(f"\n测试集准确率:{test_accuracy:.2%}")

# 7. 预测一张测试图片
image, label = test_data[0]

model.eval()
with torch.no_grad():
    image_input = image.unsqueeze(0).to(device)
    output = model(image_input)
    prediction = output.argmax(dim=1).item()

print("这张图片的真实标签：", label)
print("模型预测的数字：", prediction)
print("预测是否正确：", prediction == label)
