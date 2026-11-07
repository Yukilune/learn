import torch
import torch.nn as nn

from torchvision import datasets,transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

test_dataset = datasets.MNIST(
    root="./data",
    train = False,
    download= True,
    transform = transform
)

# 取出测试集前 10 张图片
images = torch.stack([
    test_dataset[i][0] for i in range(10)
])

# 取出对应的 10 个标签
labels = torch.tensor([
    test_dataset[i][1] for i in range(10)
])


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
state_dict = torch.load(
    "mnist_vit.pth",
    map_location= "cpu"
)

model.load_state_dict(state_dict)

model.eval()

with torch.no_grad():

    # 将 10 张图片一起送入模型
    outputs = model(images)

    # 每张图片分别选择分数最高的类别
    preds = outputs.argmax(dim=1)

    # 统计预测正确的数量
    correct = (preds == labels).sum().item()

    # 输出每张图片的预测结果
    for i in range(10):
        print(
            f"第{i+1}张图片：预测={preds[i].item()}，"
            f"真实={labels[i].item()}"
        )

    print(f"\n预测正确：{correct}/10")
    print(f"准确率：{correct/10*100:.2f}%")


