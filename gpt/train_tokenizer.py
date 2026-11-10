from collections import Counter


# ==========================================
# 1. 准备语料
# ==========================================

corpus = [
    "banana",
    "bandana",
    "banana",
    "hello",
    "hello world"
]

sequences = []

for text in corpus:
    encoded = text.encode("utf-8")
    token_ids = list(encoded)
    sequences.append(token_ids)


# ==========================================
# 2. 合并函数
# ==========================================

def merge_pair(sequence, pair, new_token):

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

num_merges = 10
merge_rules = []
for step in range(num_merges):

    pair_counts = Counter()

    for sequence in sequences:

        for i in range(len(sequence) - 1):

            pair = (sequence[i], sequence[i + 1])
            pair_counts[pair] += 1

    if not pair_counts:
        break

    best_pair = pair_counts.most_common(1)[0][0]

    new_token = 256 + step

    merge_rules.append((best_pair, new_token))

    new_sequences = []

    for sequence in sequences:

        merged = merge_pair(
            sequence,
            best_pair,
            new_token
        )

        new_sequences.append(merged)

    sequences = new_sequences


# ==========================================
# 4. 构建词表
# ==========================================

vocab = {}

# 基础字节 Token
for i in range(256):
    vocab[i] = bytes([i])

# BPE 合并 Token
for pair, new_token in merge_rules:

    vocab[new_token] = (
        vocab[pair[0]] + vocab[pair[1]]
    )


# ==========================================
# 5. encode
# ==========================================

def encode(text, merge_rules):

    encoded = text.encode("utf-8")

    token_ids = list(encoded)

    for pair, new_token in merge_rules:

        token_ids = merge_pair(
            token_ids,
            pair,
            new_token
        )

    return token_ids


# ==========================================
# 6. decode
# ==========================================

def decode(token_ids, vocab):

    decoded_bytes = b"".join(
        vocab[token_id]
        for token_id in token_ids
    )

    return decoded_bytes.decode("utf-8")


# ==========================================
# 7. 测试
# ==========================================

text = "banana"

token_ids = encode(text, merge_rules)

decoded_text = decode(token_ids, vocab)

print("训练规则：", merge_rules)
print("编码结果：", token_ids)
print("解码结果：", decoded_text)
print("是否还原成功：", text == decoded_text)