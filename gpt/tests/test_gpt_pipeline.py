
import torch
from pathlib import Path

from gpt.tokenizer.gpt_tokenizer import BPETokenizer
from gpt.model.gpt_model import GPTModel
from gpt.generation.gpt_generate import generate


# ==========================================
# 1. 路径设置
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

TOKENIZER_PATH = PROJECT_DIR / "tokenizer" / "bpe_tokenizer.json"

CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "gpt_model.pth"


# ==========================================
# 2. 加载 Tokenizer
# ==========================================

tokenizer = BPETokenizer()
tokenizer.load(str(TOKENIZER_PATH))

vocab_size = len(tokenizer.vocab)


# ==========================================
# 3. 创建并加载 GPT 模型
# ==========================================

embed_dim = 128
num_heads = 4
num_layers = 2
max_seq_len = 4

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = GPTModel(
    vocab_size,
    embed_dim,
    num_heads,
    num_layers,
    max_seq_len
).to(device)

model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )
)

model.eval()

print("GPT 模型加载成功！")


# ==========================================
# 4. 设置生成参数
# ==========================================

prompt = "深度学习"
max_new_tokens = 20


# ==========================================
# 5. 测试五种生成策略
# ==========================================

# 1. Greedy
result = generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    max_seq_len
)

print("\n【Greedy】")
print(result)


# 2. Temperature
result = generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    max_seq_len,
    do_sample=True,
    temperature=0.7
)

print("\n【Temperature】")
print(result)


# 3. Top-k
result = generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    max_seq_len,
    do_sample=True,
    top_k=5
)

print("\n【Top-k】")
print(result)


# 4. Top-p
result = generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    max_seq_len,
    do_sample=True,
    top_p=0.9
)

print("\n【Top-p】")
print(result)


# 5. Top-k + Top-p
result = generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    max_seq_len,
    do_sample=True,
    temperature=0.7,
    top_k=5,
    top_p=0.9
)

print("\n【Top-k + Top-p】")
print(result)
