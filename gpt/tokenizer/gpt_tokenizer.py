
from collections import Counter
import json

class BPETokenizer:

    # ==========================================
    # 1. 初始化
    # ==========================================

    def __init__(self, num_merges=100):

        self.num_merges = num_merges
        self.merge_rules = []
        self.vocab = {}

    # ==========================================
    # 2. 合并函数
    # ==========================================

    def merge_pair(self, sequence, pair, new_token):

        result = []
        i = 0

        while i < len(sequence):

            if (
                i < len(sequence) - 1
                and sequence[i] == pair[0]
                and sequence[i + 1] == pair[1]
            ):
                result.append(new_token)
                i += 2

            else:
                result.append(sequence[i])
                i += 1

        return result

    # ==========================================
    # 3. BPE 训练
    # ==========================================

    def train(self, corpus):

        sequences = []

        for text in corpus:
            encoded = text.encode("utf-8")
            token_ids = list(encoded)
            sequences.append(token_ids)

        self.merge_rules = []

        for step in range(self.num_merges):

            pair_counts = Counter()

            for sequence in sequences:

                for i in range(len(sequence) - 1):

                    pair = (sequence[i], sequence[i + 1])
                    pair_counts[pair] += 1

            if not pair_counts:
                break

            best_pair = pair_counts.most_common(1)[0][0]

            new_token = 256 + step

            self.merge_rules.append((best_pair, new_token))

            new_sequences = []

            for sequence in sequences:

                merged = self.merge_pair(
                    sequence,
                    best_pair,
                    new_token
                )

                new_sequences.append(merged)

            sequences = new_sequences

        self.build_vocab()

    # ==========================================
    # 4. 构建词表
    # ==========================================

    def build_vocab(self):

        self.vocab = {}

        for i in range(256):
            self.vocab[i] = bytes([i])

        for pair, new_token in self.merge_rules:

            self.vocab[new_token] = (
                self.vocab[pair[0]] + self.vocab[pair[1]]
            )
        self.eos_token = "<|endoftext|>"
        self.eos_token_id = 256 + len(self.merge_rules)

        self.vocab[self.eos_token_id] = self.eos_token.encode("utf-8")

    # ==========================================
    # 5. 编码
    # ==========================================

    def encode(self, text, add_eos=False):

        encoded = text.encode("utf-8")
        token_ids = list(encoded)

        for pair, new_token in self.merge_rules:

            token_ids = self.merge_pair(
                token_ids,
                pair,
                new_token
            )

        if add_eos:
            token_ids.append(self.eos_token_id)

        return token_ids

    # ==========================================
    # 6. 解码
    # ==========================================

    def decode(self, token_ids):

        decoded_bytes = b"".join(
            self.vocab[token_id]
            for token_id in token_ids
            if token_id != self.eos_token_id
        )

        return decoded_bytes.decode("utf-8", errors="replace")
    def save(self, path):

        data = {
            "num_merges": self.num_merges,

            "merge_rules": [
            [list(pair), new_token]
            for pair, new_token in self.merge_rules
            ]
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    def load(self, path):

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.num_merges = data["num_merges"]

        self.merge_rules = [
            (tuple(pair), new_token)
            for pair, new_token in data["merge_rules"]
        ]

        self.build_vocab()

# ==========================================
# 7. 测试
# ==========================================

if __name__ == "__main__":

    # =================================
    # 1. 准备训练语料
    # =================================

    corpus = [
        "banana",
        "bandana",
        "banana",
        "hello",
        "hello world"
    ]

    # =================================
    # 2. 训练 Tokenizer
    # =================================

    tokenizer = BPETokenizer(num_merges=10)
    tokenizer.train(corpus)

    print("合并规则：", tokenizer.merge_rules)

    # =================================
    # 3. 测试编码和解码
    # =================================

    test_texts = [
        "banana",
        "hello",
        "hello world",
        "你好，世界！",
        "I love deep learning."
    ]

    for text in test_texts:

        token_ids = tokenizer.encode(text)

        decoded_text = tokenizer.decode(token_ids)

        print("=" * 40)
        print("原始文本：", text)
        print("Token IDs：", token_ids)
        print("解码文本：", decoded_text)
        print("还原成功：", text == decoded_text)

    # =================================
    # 4. 保存 Tokenizer
    # =================================

    tokenizer.save("bpe_tokenizer.json")

    # =================================
    # 5. 创建新对象并加载
    # =================================

    tokenizer2 = BPETokenizer()
    tokenizer2.load("bpe_tokenizer.json")

    # =================================
    # 6. 测试加载后的 Tokenizer
    # =================================

    text = "banana"

    ids = tokenizer2.encode(text)
    decoded = tokenizer2.decode(ids)

    print("=" * 40)
    print("加载后的规则：", tokenizer2.merge_rules)
    print("编码结果：", ids)
    print("解码结果：", decoded)
    print("还原成功：", text == decoded)

    # 测试普通编码