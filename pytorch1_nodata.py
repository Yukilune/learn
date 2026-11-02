import torch
import torch.nn as nn

# 1. 准备数据
# x 是输入，y 是对应的真实答案
x = torch.arange(1, 11, dtype=torch.float32).reshape(-1, 1)
y = 2 * x

# 2. 划分训练集、验证集和测试集
x_train, y_train = x[:6], y[:6]
x_val, y_val = x[6:8], y[6:8]
x_test, y_test = x[8:], y[8:]

# 3. 创建模型
# 输入 1 个特征，输出 1 个预测值
model = nn.Linear(1, 1)

# 4. 创建损失函数
# 使用均方误差衡量预测值与真实答案的差距
loss_fn = nn.MSELoss()

# 5. 创建优化器
# 使用 SGD，根据梯度更新模型参数
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

# 6. 开始训练
epochs = 200

for epoch in range(epochs):

    # ===== 训练阶段 =====
    model.train()

    # 使用整个训练集计算预测结果
    train_pred = model(x_train)

    # 计算训练损失
    train_loss = loss_fn(train_pred, y_train)

    # 清空上一次留下的梯度
    optimizer.zero_grad()

    # 反向传播，计算梯度
    train_loss.backward()

    # 根据梯度更新模型参数
    optimizer.step()

    # ===== 验证阶段 =====
    model.eval()

    # 验证时不需要计算梯度
    with torch.no_grad():
        val_pred = model(x_val)
        val_loss = loss_fn(val_pred, y_val)

    # 每 20 轮打印一次损失
    if (epoch + 1) % 20 == 0:
        print(
            f"第 {epoch + 1:3d} 轮 | "
            f"训练损失：{train_loss.item():.4f} | "
            f"验证损失：{val_loss.item():.4f}"
        )

# 7. 训练结束，使用测试集评估模型
model.eval()

with torch.no_grad():
    test_pred = model(x_test)
    test_loss = loss_fn(test_pred, y_test)

print("\n===== 测试结果 =====")
print("测试集输入：", x_test.squeeze().tolist())
print("真实答案：", y_test.squeeze().tolist())
print("模型预测：", test_pred.squeeze().tolist())
print("测试损失：", test_loss.item())

# 8. 查看训练后学到的参数
print("\n===== 模型参数 =====")
print("学习到的权重：", model.weight.item())
print("学习到的偏置：", model.bias.item())