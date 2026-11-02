import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 训练数据
train_x = torch.arange(1, 9, dtype=torch.float32).reshape(-1, 1)
train_y = 2 * train_x

# 验证数据
val_x = torch.tensor([[9.0], [10.0]])
val_y = 2 * val_x

# 创建 Dataset 和 DataLoader
train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(
    train_dataset,
    batch_size=3,
    shuffle=True
)

val_dataset = TensorDataset(val_x, val_y)
val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False
)

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 100

best_val_loss = float("inf")
best_model_state = None

for epoch in range(epochs):

    # ① 训练阶段
    model.train()
    train_total_loss = 0.0
    train_total_samples = 0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()

        pred = model(batch_x)
        loss = loss_fn(pred, batch_y)

        loss.backward()
        optimizer.step()

        train_total_loss += loss.item() * batch_x.size(0)
        train_total_samples += batch_x.size(0)

    train_avg_loss = train_total_loss / train_total_samples

    # ② 验证阶段
    model.eval()
    val_total_loss = 0.0
    val_total_samples = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)

            val_total_loss += loss.item() * batch_x.size(0)
            val_total_samples += batch_x.size(0)

    val_avg_loss = val_total_loss / val_total_samples

    if val_avg_loss < best_val_loss:
        best_val_loss = val_avg_loss
        best_model_state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        }

    # ③ 每隔 10 轮打印一次
    if (epoch + 1) % 10 == 0:
        print(
            f"第 {epoch + 1:3d} 轮 | "
            f"训练 Loss: {train_avg_loss:.4f} | "
            f"验证 Loss: {val_avg_loss:.4f}"
        )
# 恢复验证 Loss 最低时的模型参数
model.load_state_dict(best_model_state)

print("最佳验证 Loss:", best_val_loss)
# 查看最终模型参数
print("weight:", model.weight.item())
print("bias:", model.bias.item())

# 测试模型对新数据的预测
model.eval()

with torch.no_grad():
    test_x = torch.tensor([[11.0], [12.0]])
    test_pred = model(test_x)

print("预测结果：", test_pred.squeeze(-1).tolist())
print("真实结果：", [22.0, 24.0])