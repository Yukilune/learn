
import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()

        # 1. CNN 图像特征提取器
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        # 2. 将图像特征映射到共享语义空间
        self.projection = nn.Linear(256, embed_dim)

    def forward(self, images):

        # images: [B, 3, 224, 224]
        features = self.backbone(images)

        # [B, 256, 1, 1] -> [B, 256]
        features = features.flatten(start_dim=1)

        # [B, 256] -> [B, embed_dim]
        image_embeddings = self.projection(features)

        return image_embeddings


if __name__ == "__main__":

    # 模拟一个 Batch 的 4 张 RGB 图片
    images = torch.randn(4, 3, 224, 224)

    model = ImageEncoder(embed_dim=256)

    image_embeddings = model(images)

    print("输入图片形状:", images.shape)
    print("图像特征形状:", image_embeddings.shape)
