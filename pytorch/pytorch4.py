
import torch
import torch.nn as nn

# 1. 准备数据
x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_train = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

x_val = torch.tensor([[5.0], [6.0]])
y_val = torch.tensor([[10.0], [12.0]])

# 2. 创建模型
model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 3. 早停参数
best_loss = float("inf")
patience = 5
counter = 0
best_state = None

for epoch in range(100):

    # 训练阶段
    model.train()
    prediction = model(x_train)
    loss = loss_fn(prediction, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 验证阶段
    model.eval()
    with torch.no_grad():
        val_prediction = model(x_val)
        val_loss = loss_fn(val_prediction, y_val).item()

    # 判断是否取得进步
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0

        # 保存当前最佳参数
        best_state = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
        }
    else:
        counter += 1

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch + 1} | "
            f"训练损失: {loss.item():.4f} | "
            f"验证损失: {val_loss:.4f} | "
            f"计数器: {counter}"
        )

    if counter >= patience:
        print("Early Stopping!")
        break

# 恢复验证集表现最好的参数
if best_state is not None:
    model.load_state_dict(best_state)

print("最佳验证损失:", best_loss)
print("最终预测:", model(torch.tensor([[7.0]])).detach())
