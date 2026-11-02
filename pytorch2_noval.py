import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 1. 准备数据
x = torch.arange(1, 11, dtype=torch.float32).reshape(-1, 1)
y = 2 * x

# 2. 创建 Dataset 和 DataLoader
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=3, shuffle=False)

# 3. 创建模型、损失函数和优化器
model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
epochs = 100

for epoch in range(epochs):
    total_loss = 0.0

    for batch_x, batch_y in loader:

        # ① 清空旧梯度
        optimizer.zero_grad()

        # ② 模型预测
        pred = model(batch_x)

        # ③ 计算损失
        loss = loss_fn(pred, batch_y)

        # ④ 反向传播
        loss.backward()

        # ⑤ 更新参数
        optimizer.step()

        # ⑥ 累加当前批次的损失
        total_loss += loss.item()

    # ⑦ 计算本轮平均 batch 损失
    avg_loss = total_loss / len(loader)

    if (epoch + 1) % 10 == 0:
        print(f"第 {epoch + 1} 轮，平均损失：{avg_loss:.4f}")