import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from BPETokenizer import BPETokenizer
import textwrap

class GPTDataset(Dataset):

    def __init__(self, tokens, block_size):

        self.tokens = tokens
        self.block_size = block_size

        self.input_ids = []
        self.target_ids = []

        for i in range(len(tokens) - block_size):

            x = tokens[i:i + block_size]
            y = tokens[i + 1:i + block_size + 1]

            self.input_ids.append(
                torch.tensor(x, dtype=torch.long)
            )

            self.target_ids.append(
                torch.tensor(y, dtype=torch.long)
            )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):

        x = self.input_ids[index]
        y = self.target_ids[index]

        return x, y
if __name__ == "__main__":

    tokens = [5, 8, 12, 16, 20, 24]

    dataset = GPTDataset(
        tokens,
        block_size=2
    )

    print("数据集长度：", len(dataset))

    for i in range(len(dataset)):

        x, y = dataset[i]

        print("样本：", i)
        print("输入：", x)
        print("目标：", y)
        print("输入形状：", x.shape)
        print("目标形状：", y.shape)
        print("-" * 30)
    loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False
    )

    for x, y in loader:

        print("Batch 输入：", x)
        print("Batch 目标：", y)

        print("输入形状：", x.shape)
        print("目标形状：", y.shape)
    print("\n========== BPE + GPTDataset 测试 ==========")

    # 1. 加载训练好的 Tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load("bpe_tokenizer.json")

    # 2. 准备一段文本
    text = textwrap.dedent("""
    I love deep learning.
    I am learning GPT.
    GPT predicts the next token.
    """).strip()

    # 3. 文本转换为 Token IDs
    tokens = tokenizer.encode(text)

    print("Token IDs：", tokens)
    print("Token 数量：", len(tokens))

    # 4. 创建 GPTDataset
    dataset = GPTDataset(
        tokens,
        block_size=4
    )

    print("数据集长度：", len(dataset))

    # 5. 创建 DataLoader
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False
    )

    # 6. 查看第一个 Batch
    for x, y in loader:
        print("输入：", x)
        print("目标：", y)
        print("输入形状：", x.shape)
        print("目标形状：", y.shape)
        break