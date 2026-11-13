
import torch

from gpt.tokenizer.gpt_tokenizer import BPETokenizer
from gpt.model.gpt_model import GPTModel


def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    max_seq_len
):
    # 1. 将 Prompt 转换成 Token ID
    prompt_ids = tokenizer.encode(prompt)

    # 2. 转换成 Tensor，增加 Batch 维度
    idx = torch.tensor([prompt_ids], dtype=torch.long)

    # 3. 切换到评估模式
    model.eval()

    # 4. 开始自回归生成
    with torch.no_grad():

        for _ in range(max_new_tokens):

            # 只保留模型允许的上下文长度
            idx_cond = idx[:, -max_seq_len:]

            # 模型预测
            outputs = model(idx_cond)

            # 取最后一个位置的预测分数
            next_token_logits = outputs[:, -1, :]

            # 贪心选择概率最高的 Token
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

    # 5. 将完整 Token 序列解码为文本
    generated_text = tokenizer.decode(idx[0].tolist())

    return generated_text


if __name__ == "__main__":

    # 加载 Tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load("bpe_tokenizer.json")

    vocab_size = len(tokenizer.vocab)

    # 模型参数必须与训练时保持一致
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

    # 加载训练好的模型参数
    model.load_state_dict(
        torch.load("gpt_model.pth", map_location="cpu")
    )

    # 生成文本
    result = generate(
        model,
        tokenizer,
        prompt="I love",
        max_new_tokens=20,
        max_seq_len=max_seq_len
    )

    print("生成结果：")
    print(result)
