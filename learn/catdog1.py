import os
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ===== 1. 基础设置 =====
DATA_DIR = r"F:\python\CNN1\catdog-cnn\data"  # 改成你真实的数据目录
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 15
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前设备：", device)

random.seed(0)
torch.manual_seed(0)


# ===== 2. 找到图片和标签 =====
def scan_data(data_dir):
    items = []
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    # 情况一：data/cat 和 data/dog 文件夹
    cat_dir = os.path.join(data_dir, "cat")
    dog_dir = os.path.join(data_dir, "dog")

    if os.path.isdir(cat_dir) and os.path.isdir(dog_dir):
        for folder, label in [(cat_dir, 0), (dog_dir, 1)]:
            for name in os.listdir(folder):
                if name.lower().endswith(extensions):
                    items.append((os.path.join(folder, name), label))

    else:
        # 情况二：图片放在同一目录，文件名以 cat 或 dog 开头
        for name in os.listdir(data_dir):
            low = name.lower()
            if not low.endswith(extensions):
                continue

            if low.startswith("cat"):
                items.append((os.path.join(data_dir, name), 0))
            elif low.startswith("dog"):
                items.append((os.path.join(data_dir, name), 1))

    if not items:
        raise ValueError("没有找到图片，请检查 DATA_DIR 和目录结构")

    labels = {label for _, label in items}
    if labels != {0, 1}:
        raise ValueError("没有同时找到猫和狗两类图片")

    return items


# ===== 3. 处理图片 =====
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
     transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])


# ===== 4. 自定义数据集 =====
class CatDogDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]

        with Image.open(path) as img:
            img = img.convert("RGB")
            img = self.transform(img)

        return img, label


# ===== 5. 划分训练集和验证集 =====
items = scan_data(DATA_DIR)
random.shuffle(items)

split = int(len(items) * 0.85)
train_items = items[:split]
val_items = items[split:]

print("总图片数：", len(items))
print("训练集：", len(train_items))
print("验证集：", len(val_items))

train_data = CatDogDataset(train_items, train_transform)
val_data = CatDogDataset(val_items, val_transform)

train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    val_data, batch_size=BATCH_SIZE, shuffle=False
)


# ===== 6. 定义 CNN =====
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# ===== 7. 训练 =====
best_val_acc = 0.0
for epoch in range(EPOCHS):
    model.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()
        total_samples += batch_size

    train_loss = total_loss / total_samples
    train_acc = total_correct / total_samples

    # ===== 8. 验证 =====
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_catdog_model.pth")
        print(f"保存最优模型！验证准确率：{best_val_acc:.2%}")
    print(
        f"第 {epoch + 1}/{EPOCHS} 轮 | "
        f"训练损失：{train_loss:.4f} | "
        f"训练准确率：{train_acc:.2%} | "
        f"验证准确率：{val_acc:.2%}"
    )

torch.save(model.state_dict(), "catdog_model.pth")
print("模型已保存")