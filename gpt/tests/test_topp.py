
import torch
import torch.nn.functional as F

# 假设 GPT 输出的 logits
logits = torch.tensor([4.0, 2.5, 1.5, 1.0, 0.5, 0.1])

top_p = 0.8

# 1. Softmax 转换为概率
probs = F.softmax(logits, dim=-1)

# 2. 按概率从大到小排序
sorted_probs, sorted_indices = torch.sort(
    probs,
    descending=True
)

print("排序后的概率:", sorted_probs)
print("排序后的索引:", sorted_indices)

# 3. 计算累计概率
cumulative_probs = torch.cumsum(
    sorted_probs,
    dim=-1
)

print("累计概率:", cumulative_probs)

# 4. 找出需要过滤的位置
sorted_indices_to_remove = cumulative_probs > top_p

# 5. 让第一个达到阈值的 Token 保留下来
sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()

sorted_indices_to_remove[0] = False

print("需要过滤的位置:", sorted_indices_to_remove)

# 6. 将排序后的过滤结果映射回原始位置
indices_to_remove = torch.zeros_like(
    probs,
    dtype=torch.bool
)

indices_to_remove[sorted_indices] = sorted_indices_to_remove

# 7. 被过滤的位置设置为 0
filtered_probs = probs.masked_fill(
    indices_to_remove,
    0.0
)

# 8. 重新归一化
filtered_probs = filtered_probs / filtered_probs.sum()

print("最终概率:", filtered_probs)
print("概率总和:", filtered_probs.sum().item())

# 9. 随机抽取 Token
next_token = torch.multinomial(
    filtered_probs,
    num_samples=1
)

print("抽取的 Token:", next_token)
