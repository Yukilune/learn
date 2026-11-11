import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from BPETokenizer import BPETokenizer
from GPTDataset import GPTDataset
from GPTModel import GPTModel

text = """
I love deep learning.
I am learning GPT.
GPT predicts the next token.
"""

tokenizer = BPETokenizer()
tokenizer.load("bpe_tokenizer.json")

tokens = tokenizer.encode(text)

vocab_size = len(tokenizer.vocab)

dataset = GPTDataset(
    tokens,
    block_size=4
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

embed_dim = 128
num_heads = 4
num_layers = 2
max_seq_len = 4

model = GPTModel(
    vocab_size,
    embed_dim,
    num_heads,
    num_layers,
    max_seq_len
)

# ==========================================
# 1. 创建优化器和损失函数
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)

# ==========================================
# 2. 开始训练
# ==========================================

num_epochs = 10

for epoch in range(num_epochs):

    # 切换到训练模式
    model.train()

    total_loss = 0

    # 遍历整个 DataLoader
    for x, y in loader:

        # 前向传播
        outputs = model(x)

        # 计算 Loss
        loss = criterion(
            outputs.reshape(-1, vocab_size),
            y.reshape(-1)
        )

        # 反向传播和参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累加每个 Batch 的 Loss
        total_loss += loss.item()

    # 计算当前 Epoch 的平均 Loss
    avg_loss = total_loss / len(loader)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Loss: {avg_loss:.4f}"
    )

print("GPT 训练完成！")
# 保存模型参数
torch.save(
    model.state_dict(),
    "gpt_model.pth"
)

print("模型参数保存成功！")


# ==========================================
# 3. 文本生成
# ==========================================

model.eval()

# 输入提示词
prompt = "I love"

# 转换为 Token IDs
prompt_ids = tokenizer.encode(prompt)

# 转换为 Tensor，增加 Batch 维度
idx = torch.tensor(
    [prompt_ids],
    dtype=torch.long
)

max_new_tokens = 20

# 生成阶段不需要计算梯度
with torch.no_grad():

    for _ in range(max_new_tokens):

        # 只保留模型能处理的最近上下文
        idx_cond = idx[:, -max_seq_len:]

        # 前向传播
        outputs = model(idx_cond)

        # 取最后一个位置的预测分数
        next_token_logits = outputs[:, -1, :]

        # 选择分数最高的 Token
        next_token = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True
        )

        # 拼接到原序列
        idx = torch.cat(
            [idx, next_token],
            dim=1
        )

# 转换回文本
generated_text = tokenizer.decode(
    idx[0].tolist()
)

print("生成结果：", generated_text)
