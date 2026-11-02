
import torch
import torch.nn as nn
import math

# =====================================
# 1. 基本参数
# =====================================

batch_size = 2
seq_len = 5
embed_dim = 8
num_heads = 2

torch.manual_seed(42)

# 假设输入有 2 个样本，每个样本 5 个 Token，每个 Token 8 个特征
x = torch.randn(batch_size, seq_len, embed_dim)

print("输入 X:", x.shape)


# =====================================
# 2. 生成 Q、K、V
# =====================================

W_Q = nn.Linear(embed_dim, embed_dim)
W_K = nn.Linear(embed_dim, embed_dim)
W_V = nn.Linear(embed_dim, embed_dim)

Q = W_Q(x)
K = W_K(x)
V = W_V(x)

print("Q:", Q.shape)
print("K:", K.shape)
print("V:", V.shape)


# =====================================
# 3. 手动计算 Self-Attention
# =====================================

# 计算 Q 和 K 的转置矩阵相乘
scores = torch.matmul(Q, K.transpose(-2, -1))

# 缩放，避免分数过大
scores = scores / math.sqrt(embed_dim)

print("Attention Scores:", scores.shape)


# =====================================
# 4. 构造 Causal Mask
# =====================================

# True 表示禁止关注，False 表示允许关注
causal_mask = torch.triu(
    torch.ones(seq_len, seq_len, dtype=torch.bool),
    diagonal=1
)

# 将禁止关注的位置设置为负无穷
scores = scores.masked_fill(causal_mask, float("-inf"))

print("Causal Mask:")
print(causal_mask)


# =====================================
# 5. Softmax 得到注意力权重
# =====================================

attention_weights = torch.softmax(scores, dim=-1)

print("Attention Weights:", attention_weights.shape)
print(attention_weights[0])


# =====================================
# 6. 根据权重融合 Value 信息
# =====================================

output = torch.matmul(attention_weights, V)

print("Attention Output:", output.shape)


# =====================================
# 7. 使用 nn.MultiheadAttention
# =====================================

attention = nn.MultiheadAttention(
    embed_dim=embed_dim,
    num_heads=num_heads,
    batch_first=True
)

# 使用同一个输入作为 Q、K、V
# 这就是 Self-Attention
mha_output, attention_weights = attention(
    x, x, x,
    attn_mask=causal_mask,
    average_attn_weights=False
)

print("MHA Output:", mha_output.shape)
print("MHA Attention Weights:", attention_weights.shape)


# =====================================
# 8. Cross-Attention
# =====================================

# 假设 Query 来自一个长度为 3 的序列
query = torch.randn(batch_size, 3, embed_dim)

# Key 和 Value 来自另一个长度为 5 的序列
key = torch.randn(batch_size, 5, embed_dim)
value = torch.randn(batch_size, 5, embed_dim)

# Cross-Attention 的位置掩码
cross_mask = torch.zeros(3, 5, dtype=torch.bool)

# 禁止第二个 Query 关注第五个 Key
cross_mask[1, 4] = True

cross_output, cross_weights = attention(
    query,
    key,
    value,
    attn_mask=cross_mask,
    average_attn_weights=False
)

print("Cross-Attention Output:", cross_output.shape)
print("Cross-Attention Weights:", cross_weights.shape)


# =====================================
# 9. Key Padding Mask
# =====================================

# Batch 中第二个样本的最后两个 Key 是 PAD
key_padding_mask = torch.tensor([
    [False, False, False, False, False],
    [False, False, False, True, True]
])

padding_output, padding_weights = attention(
    query,
    key,
    value,
    key_padding_mask=key_padding_mask,
    average_attn_weights=False
)

print("Padding Output:", padding_output.shape)
print("Padding Weights:", padding_weights.shape)
