
import torch
import torch.nn.functional as F


def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    max_seq_len,
    do_sample=False,
    temperature=1.0,
    top_k=None,
    top_p=None
):
    # ==========================================
    # 1. 初始化
    # ==========================================

    # 设置模型为推理模式
    model.eval()

    # 获取模型所在设备
    device = next(model.parameters()).device

    # 将 Prompt 转换为 Token ID
    prompt_ids = tokenizer.encode(prompt)

    # 增加 Batch 维度
    idx = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device
    )

    # ==========================================
    # 2. 自回归生成
    # ==========================================

    with torch.no_grad():

        for _ in range(max_new_tokens):

            # 限制输入上下文长度
            idx_cond = idx[:, -max_seq_len:]

            # 模型前向传播
            outputs = model(idx_cond)

            # 只取最后一个位置的预测结果
            next_token_logits = outputs[:, -1, :]

            # ==================================
            # 方式一：Greedy Decoding
            # ==================================

            if not do_sample:

                next_token = torch.argmax(
                    next_token_logits,
                    dim=-1,
                    keepdim=True
                )

            # ==================================
            # 方式二：Sampling
            # ==================================

            else:

                logits = next_token_logits.clone()

                # Temperature
                if temperature <= 0:
                    raise ValueError("temperature 必须大于 0")

                logits = logits / temperature

                # ------------------------------
                # Top-k Sampling
                # ------------------------------

                if top_k is not None:

                    if top_k < 1:
                        raise ValueError("top_k 必须至少为 1")

                    k = min(top_k, logits.size(-1))

                    top_k_values, _ = torch.topk(
                        logits,
                        k=k,
                        dim=-1
                    )

                    threshold = top_k_values[:, -1].unsqueeze(-1)

                    logits = logits.masked_fill(
                        logits < threshold,
                        float("-inf")
                    )

                # ------------------------------
                # Top-p Sampling
                # ------------------------------

                if top_p is not None:

                    if not 0 < top_p <= 1:
                        raise ValueError(
                            "top_p 必须在 (0, 1] 范围内"
                        )

                    # 按 Logits 从大到小排序
                    sorted_logits, sorted_indices = torch.sort(
                        logits,
                        descending=True,
                        dim=-1
                    )

                    # 计算概率
                    sorted_probs = F.softmax(
                        sorted_logits,
                        dim=-1
                    )

                    # 累计概率
                    cumulative_probs = torch.cumsum(
                        sorted_probs,
                        dim=-1
                    )

                    # 找出累计概率超过 top_p 的位置
                    sorted_indices_to_remove = (
                        cumulative_probs > top_p
                    )

                    # 向右移动一位，保留第一个超过阈值的 Token
                    sorted_indices_to_remove[..., 1:] = (
                        sorted_indices_to_remove[..., :-1].clone()
                    )

                    # 至少保留概率最大的 Token
                    sorted_indices_to_remove[..., 0] = False

                    # 将排序后的过滤标记映射回原始词表位置
                    indices_to_remove = torch.zeros_like(
                        logits,
                        dtype=torch.bool
                    )

                    indices_to_remove.scatter_(
                        dim=-1,
                        index=sorted_indices,
                        src=sorted_indices_to_remove
                    )

                    # 删除不符合条件的 Token
                    logits = logits.masked_fill(
                        indices_to_remove,
                        float("-inf")
                    )

                # ------------------------------
                # 根据最终概率分布进行采样
                # ------------------------------

                probs = F.softmax(
                    logits,
                    dim=-1
                )

                next_token = torch.multinomial(
                    probs,
                    num_samples=1
                )

            # ==================================
            # 3. 拼接新 Token
            # ==================================

            # 注意：这里位于两个解码分支之外
            # Greedy 和 Sampling 都需要执行
            idx = torch.cat(
                [idx, next_token],
                dim=1
            )

            # ==================================
            # 4. 检查 EOS
            # ==================================

            if next_token.item() == tokenizer.eos_token_id:
                break

    # ==========================================
    # 5. 解码并返回文本
    # ==========================================

    generated_text = tokenizer.decode(
        idx[0].tolist()
    )

    return generated_text

