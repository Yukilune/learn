"""PyTorch TinyVLA：让指令选择像素，再融合执行器状态预测动作块。"""

import torch
from torch import Tensor, nn

from data import HORIZON, INSTRUCTIONS, MAX_STEP, PIXEL_XY


class TinyVLA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(len(INSTRUCTIONS), 3))
        # 坐标随模型移动设备。它由网格生成，无需存进 checkpoint。
        self.register_buffer("pixel_xy", torch.from_numpy(PIXEL_XY), persistent=False)
        self.action_head = nn.Sequential(
            nn.Linear(4, 48),
            nn.ReLU(),
            nn.Linear(48, HORIZON * 2),
            nn.Tanh(),
        )

    def forward(self, image: Tensor, token: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        """接收 [B,H,W,3]、[B]、[B,2]，返回 [B,4,2] 动作和 [B,2] 关注位置。"""
        rgb = image.flatten(start_dim=1, end_dim=2)  # [B, P, 3]，P = H × W
        query = self.queries[token]                  # [B, 3]
        scores = torch.einsum("bpc,bc->bp", rgb, query)

        # 只给非黑像素分配注意力。全黑时分数相同，softmax 回到均匀分布。
        foreground = rgb.ne(0).any(dim=-1)
        scores = scores.masked_fill(~foreground, torch.finfo(scores.dtype).min)
        attention = scores.softmax(dim=-1)           # [B, P]，每张图单独归一化
        attended_xy = attention @ self.pixel_xy      # [B, P] @ [P, 2] → [B, 2]

        features = torch.cat((attended_xy, state), dim=-1)  # [B, 4]
        actions = self.action_head(features).unflatten(dim=-1, sizes=(HORIZON, 2))
        return actions * MAX_STEP, attended_xy
