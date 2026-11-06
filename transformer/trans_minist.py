import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 1. 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# 2. 加载数据集
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)


# 3. DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=7, embed_dim=128):
        super().__init__()

        self.patch_size = patch_size

        self.proj = nn.Linear(
            patch_size * patch_size,
            embed_dim
        )

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

    def forward(self, x):

        B, C, H, W = x.shape

        x = x.unfold(2, self.patch_size, self.patch_size)
        x = x.unfold(3, self.patch_size, self.patch_size)

        x = x.permute(0, 2, 3, 1, 4, 5)

        x = x.reshape(
            B, -1,
            C * self.patch_size * self.patch_size
        )

        x = self.proj(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)

        return x

class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            patch_size = 7,
            embed_dim = 128
        )
        self.pos_embedding = nn.Parameter(
            torch.zeros(1,17,128)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = 128,
            nhead = 4,
            dim_feedforward = 256,
            batch_first=True
            )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = 2
        )
        self.head = nn.Linear(128,10)
    def forward(self,x):
        x = self.patch_embedding(x)

        x = self.pos_embedding + x

        x =self.transformer(x)

        x = x[:,0,:]

        x = self.head(x)
        return x
model = ViT()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    # 一个 Epoch 的所有 Batch 训练完后，再统计
    epoch_loss = running_loss / len(train_loader)
    accuracy = correct / total * 100

    print(
        f"Epoch [{epoch+1}/{num_epochs}], "
        f"Loss: {epoch_loss:.4f}, "
        f"Accuracy: {accuracy:.2f}%"
    )

model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():
    for images,labels in test_loader:

        output = model(images)

        preds = outputs.argmax(dim = 1)

        print("images:", images.shape)
        print("labels:", labels.shape)
        print("outputs:", outputs.shape)
        print("preds:", preds.shape)

        test_correct += (preds == labels).sum().item()

        test_total += labels.size(0)
test_accuracy = test_correct / test_total * 100

print(f"Test Accuracy:{test_accuracy:.2f}%")