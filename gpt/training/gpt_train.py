
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader

from gpt.tokenizer.gpt_tokenizer import BPETokenizer
from gpt.data.gpt_dataset import GPTDataset
from gpt.model.gpt_model import GPTModel


# ==========================================
# 1. 设置文件路径
# ==========================================

# 当前项目目录：gpt/
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Tokenizer 文件
TOKENIZER_FILE = PROJECT_DIR / "tokenizer" / "bpe_tokenizer.json"

# 训练语料文件
TRAIN_FILE = PROJECT_DIR / "data" / "train.txt"

# 模型保存位置
CHECKPOINT_FILE = PROJECT_DIR / "checkpoints" / "gpt_model.pth"

# 确保 checkpoints 文件夹存在
CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. 初始化 Tokenizer
# ==========================================

tokenizer = BPETokenizer()

tokenizer.load(str(TOKENIZER_FILE))

print("Tokenizer 加载成功！")


# ==========================================
# 3. 读取训练语料
# ==========================================

text = TRAIN_FILE.read_text(encoding="utf-8").strip()

if not text:
    raise ValueError("训练语料为空，请检查 train.txt！")


# ==========================================
# 4. 按照空行分割独立文章
# ==========================================

documents = text.split("\n\n")

# 过滤空文章
documents = [
    document.strip()
    for document in documents
    if document.strip()
]

print(f"文章数量: {len(documents)}")


# ==========================================
# 5. 逐篇文章编码，并添加 EOS
# ==========================================

tokens = []

for document in documents:

    document_tokens = tokenizer.encode(
        document,
        add_eos=True
    )

    tokens.extend(document_tokens)


# 检查编码结果
eos_count = tokens.count(tokenizer.eos_token_id)

print(f"Token 总数: {len(tokens)}")
print(f"EOS Token ID: {tokenizer.eos_token_id}")
print(f"EOS 数量: {eos_count}")

if eos_count != len(documents):
    raise ValueError("EOS 数量与文章数量不一致！")


# ==========================================
# 6. 创建 Dataset 和 DataLoader
# ==========================================

block_size = 4
batch_size = 2

dataset = GPTDataset(
    tokens,
    block_size=block_size
)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)


# ==========================================
# 7. 初始化 GPT 模型
# ==========================================

vocab_size = len(tokenizer.vocab)

embed_dim = 128
num_heads = 4
num_layers = 2
max_seq_len = block_size

model = GPTModel(
    vocab_size,
    embed_dim,
    num_heads,
    num_layers,
    max_seq_len
)


# ==========================================
# 8. 设置训练设备
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)

print(f"使用设备: {device}")


# ==========================================
# 9. 设置损失函数和优化器
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)


# ==========================================
# 10. 开始训练
# ==========================================

num_epochs = 10

for epoch in range(num_epochs):

    model.train()

    total_loss = 0

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        # 前向传播
        outputs = model(x)

        # 计算损失
        loss = criterion(
            outputs.reshape(-1, vocab_size),
            y.reshape(-1)
        )

        # 反向传播
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Loss: {avg_loss:.4f}"
    )


# ==========================================
# 11. 保存模型
# ==========================================

print("GPT 训练完成！")

torch.save(
    model.state_dict(),
    CHECKPOINT_FILE
)

print("模型参数保存成功！")
print(f"保存位置: {CHECKPOINT_FILE}")
