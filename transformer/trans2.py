
import torch
import torch.nn as nn


# ==========================================
# 1. 位置编码（这里使用可学习的位置 Embedding）
# ==========================================

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, embed_dim):
        super().__init__()

        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        positions = torch.arange(seq_len, device=x.device)

        position = self.position_embedding(positions)

        return x + position.unsqueeze(0)


# ==========================================
# 2. Encoder Layer
# ==========================================

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()

        # Self-Attention
        self.norm1 = nn.LayerNorm(embed_dim)

        self.self_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True
        )

        # FFN
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):

        # ① Self-Attention
        normalized_x = self.norm1(x)

        attention_output, _ = self.self_attn(
            normalized_x,
            normalized_x,
            normalized_x
        )

        x = x + attention_output

        # ② FFN
        normalized_x = self.norm2(x)

        ffn_output = self.ffn(normalized_x)

        x = x + ffn_output

        return x


# ==========================================
# 3. Decoder Layer
# ==========================================

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()

        # Masked Self-Attention
        self.norm1 = nn.LayerNorm(embed_dim)

        self.self_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True
        )

        # Cross-Attention
        self.norm2 = nn.LayerNorm(embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True
        )

        # FFN
        self.norm3 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x, encoder_output, causal_mask):

        # ① Masked Self-Attention
        normalized_x = self.norm1(x)

        attention_output, _ = self.self_attn(
            normalized_x,
            normalized_x,
            normalized_x,
            attn_mask=causal_mask
        )

        x = x + attention_output

        # ② Cross-Attention
        normalized_x = self.norm2(x)

        cross_output, _ = self.cross_attn(
            normalized_x,
            encoder_output,
            encoder_output
        )

        x = x + cross_output

        # ③ FFN
        normalized_x = self.norm3(x)

        ffn_output = self.ffn(normalized_x)

        x = x + ffn_output

        return x


# ==========================================
# 4. 完整 Encoder
# ==========================================

class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len,
        embed_dim,
        num_heads,
        hidden_dim,
        num_layers
    ):
        super().__init__()

        # Token Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional Encoding
        self.position = PositionalEncoding(max_len, embed_dim)

        # 多层 Encoder Layer
        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, hidden_dim)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, src):

        # ① Token Embedding
        x = self.embedding(src)

        # ② 位置编码
        x = self.position(x)

        # ③ 依次经过多个 Encoder Layer
        for layer in self.layers:
            x = layer(x)

        # ④ 最终归一化
        encoder_output = self.final_norm(x)

        return encoder_output


# ==========================================
# 5. 完整 Decoder
# ==========================================

class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len,
        embed_dim,
        num_heads,
        hidden_dim,
        num_layers
    ):
        super().__init__()

        # Token Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional Encoding
        self.position = PositionalEncoding(max_len, embed_dim)

        # 多层 Decoder Layer
        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, hidden_dim)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, tgt, encoder_output):

        # ① Token Embedding
        x = self.embedding(tgt)

        # ② 位置编码
        x = self.position(x)

        # ③ 创建 Causal Mask
        seq_len = tgt.size(1)

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=tgt.device
            ),
            diagonal=1
        )

        # ④ 依次经过多个 Decoder Layer
        for layer in self.layers:
            x = layer(x, encoder_output, causal_mask)

        # ⑤ 最终归一化
        decoder_output = self.final_norm(x)

        return decoder_output


# ==========================================
# 6. 完整 Transformer
# ==========================================

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        max_len,
        embed_dim,
        num_heads,
        hidden_dim,
        num_layers
    ):
        super().__init__()

        self.encoder = Encoder(
            src_vocab_size,
            max_len,
            embed_dim,
            num_heads,
            hidden_dim,
            num_layers
        )

        self.decoder = Decoder(
            tgt_vocab_size,
            max_len,
            embed_dim,
            num_heads,
            hidden_dim,
            num_layers
        )

        # 将 Decoder 特征映射到目标词表
        self.output_linear = nn.Linear(embed_dim, tgt_vocab_size)

    def forward(self, src, tgt):

        # ① Encoder 处理源序列
        encoder_output = self.encoder(src)

        # ② Decoder 结合 Encoder Output 处理目标序列
        decoder_output = self.decoder(tgt, encoder_output)

        # ③ 预测每个位置的词表分数
        logits = self.output_linear(decoder_output)

        return logits


# ==========================================
# 7. 测试完整 Transformer
# ==========================================

if __name__ == "__main__":

    # 基本参数
    batch_size = 2

    src_seq_len = 5
    tgt_seq_len = 4

    src_vocab_size = 100
    tgt_vocab_size = 120

    embed_dim = 16
    num_heads = 4
    hidden_dim = 64
    num_layers = 2
    max_len = 50

    # 模拟源语言 Token ID
    src = torch.randint(
        0,
        src_vocab_size,
        (batch_size, src_seq_len)
    )

    # 模拟目标语言 Token ID
    tgt = torch.randint(
        0,
        tgt_vocab_size,
        (batch_size, tgt_seq_len)
    )

    # 创建模型
    model = Transformer(
        src_vocab_size,
        tgt_vocab_size,
        max_len,
        embed_dim,
        num_heads,
        hidden_dim,
        num_layers
    )

    # 前向传播
    logits = model(src, tgt)

    # 查看结果
    print("源序列形状：", src.shape)
    print("目标序列形状：", tgt.shape)
    print("预测结果形状：", logits.shape)

    print("模型运行成功！")
