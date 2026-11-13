import torch
import torch.nn.functional as F

# 假设 GPT 输出的 logits
logits = torch.tensor([2.0, 1.0, 0.1])

# 设置不同温度
temperatures = [0.5, 1.0, 2.0]

for temperature in temperatures:

    # 调整 logits
    scaled_logits = logits / temperature

    # 转换成概率
    probs = F.softmax(scaled_logits, dim=-1)

    print(f"\nTemperature: {temperature}")
    print("调整后的 logits:", scaled_logits)
    print("概率分布:", probs)
    print("概率总和:", probs.sum().item())