
import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size=1000,
        embed_dim=256,
        max_len=32,
        num_heads=4,
        num_layers=2,
        pad_id=0
    ):
        super().__init__()

        self.pad_id = pad_id

        # 1. Token Embedding
        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_id
        )

        # 2. Position Embedding
        self.position_embedding = nn.Embedding(
            max_len,
            embed_dim
        )

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 4. Projection
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, token_ids):

        # token_ids: [B, L]
        batch_size, seq_len = token_ids.shape

        # 生成位置索引 [L]
        positions = torch.arange(
            seq_len,
            device=token_ids.device
        )

        # Token Embedding: [B, L] -> [B, L, 256]
        token_features = self.token_embedding(token_ids)

        # Position Embedding: [L] -> [L, 256]
        position_features = self.position_embedding(positions)

        # 广播相加: [B, L, 256] + [L, 256]
        x = token_features + position_features

        # 标记 PAD 位置，True 表示忽略
        padding_mask = token_ids.eq(self.pad_id)

        # Transformer: [B, L, 256]
        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask
        )

        # 只对非 PAD 位置求平均
        valid_mask = (~padding_mask).unsqueeze(-1)

        x = x * valid_mask

        valid_count = valid_mask.sum(dim=1).clamp(min=1)

        # [B, L, 256] -> [B, 256]
        text_features = x.sum(dim=1) / valid_count

        # Projection: [B, 256] -> [B, 256]
        text_embeddings = self.projection(text_features)

        return text_embeddings


if __name__ == "__main__":

    # 模拟 3 条句子，每条句子长度为 6
    token_ids = torch.tensor([
        [12, 8, 5, 21, 2, 0],
        [12, 9, 5, 17, 2, 0],
        [12, 10, 5, 18, 2, 0]
    ])

    model = TextEncoder(
        vocab_size=1000,
        embed_dim=256,
        max_len=32
    )

    text_embeddings = model(token_ids)

    print("输入 Token ID 形状:", token_ids.shape)
    print("文本特征形状:", text_embeddings.shape)
