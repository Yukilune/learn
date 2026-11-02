
import torch
import torch.nn as nn

torch.manual_seed(0)

# 1. 准备数据：前三天预测第四天
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
    [23.0], [24.0], [25.0], [26.0],
    [27.0], [28.0], [29.0], [30.0]
])
X = (X - 20.0) / 10.0
y = (y - 20.0) / 10.0

# 2. 定义 RNN
class TemperatureRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=1,
            hidden_size=8,
            batch_first=True
        )
        self.fc = nn.Linear(8, 1)

    def forward(self, x):
        output, (h_n, c_n) = self.rnn(x)

        last_output = output[:, -1, :]
        prediction = self.fc(last_output)

        return prediction

model = TemperatureRNN()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.01
)

# 3. 记录训练前的一个权重
weight_before = (
    model.rnn.weight_ih_l0[0, 0].item()
)

# 4. 先观察训练前的 loss
model.eval()
with torch.no_grad():
    prediction = model(X)
    loss_before = loss_fn(prediction, y).item()

print(f"训练前 loss：{loss_before:.4f}")
print(f"训练前 RNN 权重：{weight_before:.6f}")

# 5. 开始训练
for epoch in range(201):
    model.train()

    prediction = model(X)
    loss = loss_fn(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    print(
    "LSTM 输入权重梯度：",
    model.rnn.weight_ih_l0.grad.norm().item()
    )
    optimizer.step()

    if epoch % 50 == 0:
        print(
            f"Epoch {epoch:3d} | "
            f"Loss: {loss.item():.4f}"
        )

# 6. 再观察训练后的 loss 和权重

# 6. 查看训练后的 loss、权重和预测温度
model.eval()

with torch.no_grad():
    prediction = model(X)
    loss_after = loss_fn(prediction, y).item()

    # 将归一化后的预测值转换回摄氏度
    predicted_temp = prediction * 10 + 20

print("\n训练后 loss：", loss_after)
print("预测温度：")
print(predicted_temp.flatten())

weight_after = model.rnn.weight_ih_l0[0, 0].item()

print("训练后 RNN 权重：", weight_after)
print("权重变化量：", weight_after - weight_before)
# 7. 用训练时没有出现过的新序列进行预测
new_data = torch.tensor([[
    [0.8],
    [0.9],
    [1.0]
]])



with torch.no_grad():
    new_prediction = model(new_data)

# 还原为摄氏度
predicted_temp = new_prediction.item() * 10 + 20

print(f"新序列预测温度：{predicted_temp:.2f}℃")
# 测试不同的新序列
test_sequences = [
    [0.9, 1.0, 1.1],  # 29、30、31℃
    [1.0, 1.1, 1.2],  # 30、31、32℃
]



with torch.no_grad():
    for seq in test_sequences:
        new_data = torch.tensor(
            [[ [v] for v in seq ]],
            dtype=torch.float32
        )

        prediction = model(new_data)
        predicted_temp = prediction.item() * 10 + 20

        print(
            f"输入：{[v * 10 + 20 for v in seq]}℃，"
            f"预测：{predicted_temp:.2f}℃"
        )