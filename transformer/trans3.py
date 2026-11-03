
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. 位置编码
# ==========================================

class PositionalEncoding(nn.Module):

    def __init__(self, vocab_size, embed_dim, max_len):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size, embed_dim
        )

        self.position_embedding = nn.Embedding(
            max_len, embed_dim
        )

    def forward(self, x):

        batch_size, seq_len = x.shape

        positions = torch.arange(
            seq_len, device=x.device
        )

        token_x = self.token_embedding(x)
        position_x = self.position_embedding(positions)

        return token_x + position_x


# ==========================================
# 2. Encoder Layer
# ==========================================

class EncoderLayer(nn.Module):

    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):

        # Self-Attention
        normalized_x = self.norm1(x)

        attn_output, _ = self.self_attn(
            normalized_x,
            normalized_x,
            normalized_x,
            need_weights=False
        )

        x = x + attn_output

        # FFN
        ffn_output = self.ffn(self.norm2(x))

        x = x + ffn_output

        return x


# ==========================================
# 3. Decoder Layer
# ==========================================

class DecoderLayer(nn.Module):

    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()

        # Masked Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True
        )

        # Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x, encoder_output):

        seq_len = x.size(1)

        # Causal Mask：禁止关注未来位置
        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=x.device
            ),
            diagonal=1
        )

        # ① Masked Self-Attention
        normalized_x = self.norm1(x)

        self_output, _ = self.self_attn(
            normalized_x,
            normalized_x,
            normalized_x,
            attn_mask=causal_mask,
            need_weights=False
        )

        x = x + self_output

        # ② Cross-Attention
        normalized_x = self.norm2(x)

        cross_output, _ = self.cross_attn(
            normalized_x,       # Query：来自 Decoder
            encoder_output,     # Key：来自 Encoder
            encoder_output,     # Value：来自 Encoder
            need_weights=False
        )

        x = x + cross_output

        # ③ FFN
        ffn_output = self.ffn(self.norm3(x))

        x = x + ffn_output

        return x


# ==========================================
# 4. Encoder
# ==========================================

class Encoder(nn.Module):

    def __init__(
        self,
        src_vocab_size,
        embed_dim,
        num_heads,
        hidden_dim,
        num_layers,
        max_len
    ):
        super().__init__()

        self.embedding = PositionalEncoding(
            src_vocab_size,
            embed_dim,
            max_len
        )

        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, hidden_dim)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, src):

        x = self.embedding(src)

        for layer in self.layers:
            x = layer(x)

        return self.norm(x)


# ==========================================
# 5. Decoder
# ==========================================

class Decoder(nn.Module):

    def __init__(
        self,
        tgt_vocab_size,
        embed_dim,
        num_heads,
        hidden_dim,
        num_layers,
        max_len
    ):
        super().__init__()

        self.embedding = PositionalEncoding(
            tgt_vocab_size,
            embed_dim,
            max_len
        )

        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, hidden_dim)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tgt, encoder_output):

        x = self.embedding(tgt)

        for layer in self.layers:
            x = layer(x, encoder_output)

        return self.norm(x)


# ==========================================
# 6. 完整 Transformer
# ==========================================

class Transformer(nn.Module):

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        embed_dim=128,
        num_heads=4,
        hidden_dim=256,
        num_layers=2,
        max_len=100
    ):
        super().__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_dim,
            num_heads,
            hidden_dim,
            num_layers,
            max_len
        )

        self.decoder = Decoder(
            tgt_vocab_size,
            embed_dim,
            num_heads,
            hidden_dim,
            num_layers,
            max_len
        )

        # 将隐藏特征映射为目标词表分数
        self.output_linear = nn.Linear(
            embed_dim,
            tgt_vocab_size
        )

    def forward(self, src, tgt):

        # Encoder 处理英文
        encoder_output = self.encoder(src)

        # Decoder 处理中文前缀
        decoder_output = self.decoder(
            tgt,
            encoder_output
        )

        # 输出每个位置的词表分数
        logits = self.output_linear(decoder_output)

        return logits


# ==========================================
# 7. 创建模型
# ==========================================

src_vocab_size = 100
tgt_vocab_size = 100

BOS_ID = 1
EOS_ID = 2

model = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size
)


# ==========================================
# 8. 训练阶段：Teacher Forcing
# ==========================================

# 假设英文输入有两个样本
src = torch.tensor([
    [10, 11, 12],
    [13, 14, 15]
])

# 中文标准答案：我 爱 你 EOS
tgt = torch.tensor([
    [20, 21, 22, EOS_ID],
    [23, 24, 25, EOS_ID]
])

# Decoder 输入：右移一位
tgt_input = torch.cat([
    torch.full((tgt.size(0), 1), BOS_ID),
    tgt[:, :-1]
], dim=1)

# 模型前向传播
logits = model(src, tgt_input)

print("训练阶段 logits:", logits.shape)

# 计算交叉熵
loss = F.cross_entropy(
    logits.reshape(-1, tgt_vocab_size),
    tgt.reshape(-1)
)

print("Loss:", loss.item())


# ==========================================
# 9. 自回归生成阶段
# ==========================================

@torch.no_grad()
def generate(model, src, max_new_tokens=10):

    model.eval()

    # Encoder 处理源句
    encoder_output = model.encoder(src)

    # 第一次只输入 BOS
    generated = torch.tensor(
        [[BOS_ID]],
        device=src.device
    )

    for step in range(max_new_tokens):

        # Decoder 根据当前已有的 Token 预测
        decoder_output = model.decoder(
            generated,
            encoder_output
        )

        # 映射到目标词表
        logits = model.output_linear(decoder_output)

        # 只取最后一个位置的分数
        next_token_logits = logits[:, -1, :]

        # 转换为概率
        probabilities = F.softmax(
            next_token_logits,
            dim=-1
        )

        # 选择概率最高的 Token
        next_token = probabilities.argmax(
            dim=-1,
            keepdim=True
        )

        print(
            f"第 {step + 1} 次生成，"
            f"输入：{generated.tolist()}，"
            f"新 Token：{next_token.item()}"
        )

        # 拼接到已有序列
        generated = torch.cat(
            [generated, next_token],
            dim=1
        )

        # 遇到结束标记，停止生成
        if next_token.item() == EOS_ID:
            break

    return generated


# ==========================================
# 10. 测试生成
# ==========================================

test_src = torch.tensor([[10, 11, 12]])

result = generate(model, test_src)

print("最终生成结果：", result)
