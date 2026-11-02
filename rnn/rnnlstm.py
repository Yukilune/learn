import torch
import torch.nn as nn

torch.manual_seed(0)


class SimpleLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.linear = nn.Linear(
            input_size + hidden_size,
            4 * hidden_size
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)

        gates = self.linear(combined)
        f, i, g, o = gates.chunk(4, dim=1)

        f = torch.sigmoid(f)
        i = torch.sigmoid(i)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)

        return h_new, c_new, f


cell = SimpleLSTMCell(input_size=1, hidden_size=2)

optimizer = torch.optim.SGD(cell.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

x = torch.tensor([[0.5]])
h = torch.zeros(1, 2)
c = torch.zeros(1, 2)

# ① 复制更新前的权重
weight_before = cell.linear.weight.detach().clone()

# ② 前向计算
prediction, c_new, f = cell(x, h, c)

# ③ 假设目标输出为两个 0.5
target = torch.tensor([[0.5, 0.5]])
loss = loss_fn(prediction, target)

# ④ 反向传播并更新参数
optimizer.zero_grad()
loss.backward()
optimizer.step()

# ⑤ 查看更新后的权重
weight_after = cell.linear.weight.detach()

print("更新前的权重：")
print(weight_before)

print("\n更新后的权重：")
print(weight_after)

print("\n权重变化总量：")
print((weight_after - weight_before).norm().item())

print("\n本次更新前的损失：")
print(loss.item())