
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from clip.models.image_encoder import ImageEncoder
from clip.models.text_encoder import TextEncoder


class CLIP(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()

        # 图像编码器
        self.image_encoder = ImageEncoder(
            embed_dim=embed_dim
        )

        # 文本编码器
        self.text_encoder = TextEncoder(
            embed_dim=embed_dim
        )

        # 可学习的温度缩放参数
        
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
            

    def forward(self, images, token_ids):

        # 1. 分别提取图像和文本特征
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(token_ids)

        # 2. 对特征进行 L2 归一化
        image_features = F.normalize(
            image_features, dim=-1
        )

        text_features = F.normalize(
            text_features, dim=-1
        )

        # 3. 计算缩放系数
        scale = self.logit_scale.exp().clamp(max=100)

        # 4. 计算图像与文本之间的相似度
        logits_per_image = scale * (
            image_features @ text_features.T
        )

        # 5. 转置得到文本对图像的相似度
        logits_per_text = logits_per_image.T

        return logits_per_image, logits_per_text


def clip_loss(logits_per_image, logits_per_text):

    batch_size = logits_per_image.shape[0]

    # 正确配对的标签位于对角线
    labels = torch.arange(
        batch_size,
        device=logits_per_image.device
    )

    # 图像找文本 + 文本找图像
    loss_image = F.cross_entropy(
        logits_per_image, labels
    )

    loss_text = F.cross_entropy(
        logits_per_text, labels
    )

    # 对称对比损失
    loss = (loss_image + loss_text) / 2

    return loss


if __name__ == "__main__":

    # 模拟一个 Batch：4 张图片、4 条文本
    images = torch.randn(4, 3, 224, 224)

    token_ids = torch.randint(1, 1000, (4, 12))
    token_ids[:, -1] = 0

    model = CLIP(embed_dim=256)

    logits_per_image, logits_per_text = model(
        images, token_ids
    )

    loss = clip_loss(
        logits_per_image,
        logits_per_text
    )

    print("图像与文本相似度矩阵:")
    print(logits_per_image)

    print("图像相似度矩阵形状:", logits_per_image.shape)
    print("文本相似度矩阵形状:", logits_per_text.shape)
    print("CLIP Loss:", loss.item())
