
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ==========================================
# 1. 准备数据
# ==========================================

texts = [
    "i love this movie",
    "this movie is wonderful",
    "i really like this film",
    "the movie is amazing",
    "what a great movie",
    "i enjoy this film",
    "this is a fantastic movie",
    "the film is excellent",

    "i hate this movie",
    "this movie is terrible",
    "i really dislike this film",
    "the movie is awful",
    "what a boring movie",
    "i regret watching this film",
    "this is a horrible movie",
    "the film is disappointing"
]

# 1 表示正面，0 表示负面
labels = [
    1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0
]


# ==========================================
# 2. 分词，建立词表
# ==========================================

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


word_set = set()

for text in texts:
    word_set.update(tokenize(text))

# 0 专门留给 PAD
word2idx = {"<PAD>": 0}

for word in sorted(word_set):
    word2idx[word] = len(word2idx)

vocab_size = len(word2idx)

print("词表大小：", vocab_size)


# ==========================================
# 3. 文本转换为 Token ID
# ==========================================

max_len = max(len(tokenize(text)) for text in texts)

def encode(text):
    tokens = tokenize(text)

    ids = [word2idx[word] for word in tokens]

    # 不足 max_len 的位置补 PAD
    ids += [0] * (max_len - len(ids))

    return ids


encoded_texts = torch.tensor(
    [encode(text) for text in texts],
    dtype=torch.long
)

labels = torch.tensor(labels, dtype=torch.long)


# ==========================================
# 4. 构建 Dataset 和 DataLoader
# ==========================================

class TextDataset(Dataset):

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]


dataset = TextDataset(encoded_texts, labels)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)


# ==========================================
# 5. 位置编码
# ==========================================

class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_len):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embed_dim, 2)
            * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )

        pe = torch.zeros(max_len, embed_dim)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# ==========================================
# 6. Transformer 文本分类模型
# ==========================================

class TransformerClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        num_classes=2,
        max_len=50
    ):
        super().__init__()

        # Token Embedding
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )

        # Position Encoding
        self.position_encoding = PositionalEncoding(
            embed_dim,
            max_len
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 分类层
        self.classifier = nn.Linear(
            embed_dim,
            num_classes
        )

    def forward(self, x):

        # 找出 PAD 位置
        padding_mask = (x == 0)

        # Embedding
        x = self.embedding(x)

        # Position Encoding
        x = self.position_encoding(x)

        # Transformer Encoder
        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask
        )

        # Masked Mean Pooling
        valid_mask = (~padding_mask).unsqueeze(-1)

        x = x * valid_mask

        x = x.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)

        # 分类
        logits = self.classifier(x)

        return logits


# ==========================================
# 7. 初始化模型、损失函数和优化器
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = TransformerClassifier(
    vocab_size=vocab_size,
    embed_dim=32,
    num_heads=4,
    num_layers=2,
    num_classes=2,
    max_len=max_len
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# 8. 训练模型
# ==========================================

epochs = 100

for epoch in range(epochs):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for batch_texts, batch_labels in loader:

        batch_texts = batch_texts.to(device)
        batch_labels = batch_labels.to(device)

        # 前向传播
        logits = model(batch_texts)

        # 计算损失
        loss = criterion(logits, batch_labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 计算准确率
        predictions = logits.argmax(dim=1)

        correct += (predictions == batch_labels).sum().item()
        total += batch_labels.size(0)

    accuracy = correct / total

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1:03d} | "
            f"Loss: {total_loss / len(loader):.4f} | "
            f"Accuracy: {accuracy:.2%}"
        )


# ==========================================
# 9. 使用训练好的模型进行预测
# ==========================================

def predict(text):

    model.eval()

    with torch.no_grad():

        ids = encode(text)

        x = torch.tensor(
            [ids],
            dtype=torch.long
        ).to(device)

        logits = model(x)

        probabilities = torch.softmax(logits, dim=1)

        prediction = probabilities.argmax(dim=1).item()

        result = "正面评价" if prediction == 1 else "负面评价"

        print("输入文本：", text)
        print("预测结果：", result)
        print("类别概率：", probabilities.cpu().numpy())


predict("i love this film")
predict("this movie is terrible")
