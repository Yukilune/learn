
import torch
import torch.nn.functional as F

# 假设 GPT 输出 6 个候选 Token 的 logits
logits = torch.tensor([4.0, 2.5, 1.5, 1.0, 0.5, 0.1])

# 只保留概率最高的 3 个 Token
top_k = 3

# 获取最大的 3 个 logits
values, indices = torch.topk(
    logits,
    k=top_k
)

print("Top-k values:", values)
print("Top-k indices:", indices)

# 获取第 k 大的 logit，作为过滤阈值
threshold = values[-1]

# 把低于阈值的 logits 替换为负无穷
filtered_logits = logits.masked_fill(
    logits < threshold,
    float("-inf")
)

print("过滤后的 logits:", filtered_logits)

# Softmax 转换成概率
probs = F.softmax(filtered_logits, dim=-1)

print("最终概率:", probs)
print("概率总和:", probs.sum().item())
for i in range(10):
    # 按照概率随机抽取一个 Token
    next_token = torch.multinomial(
        probs,
        num_samples=1
    )

    print("抽取的 Token:", next_token)
