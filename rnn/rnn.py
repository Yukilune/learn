
import torch
import torch.nn as nn

torch.manual_seed(0)

# 1. 准备训练数据：用前 3 天预测第 4 天
X = torch.tensor([
    [[20.0], [21.0], [22.0]],
    [[21.0], [22.0], [23.0]],
    [[22.0], [23.0], [24.0]],
    [[23.0], [24.0], [25.0]],
    [[24.0], [25.0], [26.0]],
    [[25.0], [26.0], [27.0]],
    [[26.0], [27.0], [28.0]],
    [[27.0], [28.0], [29.0]],
])

y = torch.tensor([
    [23.0],
    [24.0],
    [25.0],
    [26.0],
    [27.0],
    [28.0],
    [29.0],
    [30.0],
])

# 2. 数据归一化：把温度缩放到较小的数值范围
X = (X - 20.0) / 10.0
y = (y - 20.0) / 10.0


# 3. 定义 RNN 模型
class TemperatureRNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=8,
            batch_first=True
        )

        self.fc = nn.Linear(8, 1)

    def forward(self, x):
        output, hidden = self.rnn(x)

        # 取最后一个时间步的输出
        last_output = output[:, -1, :]

        # 预测归一化后的第 4 天温度
        prediction = self.fc(last_output)

        return prediction


# 4. 创建模型、损失函数和优化器
model = TemperatureRNN()

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


# 5. 开始训练
for epoch in range(501):
    model.train()

    prediction = model(X)
    loss = loss_fn(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# 6. 检查模型对训练数据的预测
model.eval()

with torch.no_grad():
    predictions = model(X)

print("\n训练样本预测结果：")

for i in range(len(X)):
    # 把预测值还原成摄氏温度
    predicted_temp = predictions[i].item() * 10 + 20

    # y 也做过归一化，所以还原真实值
    true_temp = y[i].item() * 10 + 20

    print(
        f"样本{i + 1}："
        f"预测 {predicted_temp:.2f}℃，"
        f"真实值 {true_temp:.2f}℃"
    )


# 7. 使用新数据预测第 4 天温度
# 原始温度为 28℃、29℃、30℃
# 归一化后为 0.8、0.9、1.0
new_data = torch.tensor([[
    [0.8],
    [0.9],
    [1.0]
]])

with torch.no_grad():
    result = model(new_data)

# 将预测结果还原为摄氏温度
predicted_temp = result.item() * 10 + 20

print("\n新数据预测结果：")
print(f"预测第 4 天气温：{predicted_temp:.2f}℃")
print("真实规律对应的气温：31.00℃")
