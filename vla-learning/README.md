# 用一个可运行的 VLA 学习视觉、语言和动作

画面里同时有一个红色目标和一个蓝色目标。模型看到 RGB 图、文本指令和执行器的当前位置，预测接下来四步的二维位移。保持画面和初始位置相同，只换一句指令，就能检查模型是否朝另一个目标移动。

这个练习用 NumPy 生成示教和执行二维环境，用 PyTorch 在 CPU 上训练、保存和运行模型。图像尺寸为 8×8，词表只有两条固定指令。模型用监督式行为克隆，学到的能力限定在这个合成任务里。

## 跑一遍

在 PowerShell 中用已安装 PyTorch 与 NumPy 的 Python 环境执行。安装方法参见 [PyTorch 本地安装指南](https://docs.pytorch.org/get-started/)。

```powershell
cd F:\codex\vla-learning
python -m pip install -r requirements.txt
python lesson.py inspect
python lesson.py train
python lesson.py evaluate
python lesson.py demo --instruction "move to the red target"
python lesson.py demo --instruction "move to the blue target"
```

`inspect` 展示两条示教样本的状态和动作标签。`train` 把 PyTorch `state_dict` 保存到 `runs/tiny_vla.pt`。`evaluate` 重新加载权重，比较专家、完整输入与三种输入干预。`demo` 打印模型关注的位置、第一段动作预测，以及执行过程的坐标。`python lesson.py evaluate --execute 4` 可以比较每次执行两步或四步时的闭环结果。

如果当前目录已有本机验证时用的 `.venv-vla`，可直接用 `.\.venv-vla\Scripts\python.exe` 代替命令里的 `python`。

已有 NumPy 权重还可以用 `python lesson.py --checkpoint runs/tiny_vla.npz evaluate` 评估。加载时会把旧矩阵从 `[in, out]` 转为 `nn.Linear` 使用的 `[out, in]`。

## 第一课：示教如何变成训练样本

读 [`data.py`](data.py) 的 `expert_episodes()` 和 `training_windows()`。专家知道目标坐标，每步将目标与当前位置的差乘以 0.5，并限制每轴最大位移。每条示教有 8 个动作和 9 个状态。模型每次用 `state[t]`，预测 `actions[t:t+4]`。最后三个起点无法提供完整的四步标签，所以这个例子每条示教只取五个窗口。

| 张量 | 批次形状 | 含义 |
| --- | --- | --- |
| `image` | `[B, 8, 8, 3]` | 一张 RGB 图，位置由像素索引提供 |
| `token` | `[B]` | 固定指令转为 ID，红为 `0`，蓝为 `1` |
| `state` | `[B, 2]` | 当前 `[x, y]`，坐标范围 `[-1, 1]` |
| `expert_chunk` | `[B, 4, 2]` | 专家从当前时刻起执行的四步 `[dx, dy]` |

目标坐标只给专家和评估器。先按一对目标的位置划分训练、验证与测试集合，再从各自的 episode 取窗口。因此，整条轨迹及同布局的两种指令都留在同一个集合内。`window_dataset()` 将数组转成 Tensor 并包装进 `TensorDataset`。指令 ID 显式用 `torch.long`，供后续索引指令 Query。

## 第二课：语言如何改变对图像的关注

读 [`model.py`](model.py) 的 `TinyVLA.forward()`。`nn.Parameter` 保存两条指令的可学习 Query，每条一个三维向量。对每个 RGB 像素与对应指令 Query 做点积，得到 `[B, P]` 分数，`P = H × W`。沿像素维做 softmax 后，权重与 `[P, 2]` 的坐标表相乘，汇总成 `[B, 2]` 的 `attended_xy`。坐标表注册为 buffer，随模型移动设备。

```text
指令 ID → 可学习 query ─┐
                       ├→ 像素分数 → softmax → 加权像素坐标
RGB 图 → 像素 RGB ──────┘                       │
当前状态 ───────────────────────────────────────┴→ MLP → 4 × 2 动作
```

初始 Query 都是零，两个前景像素的权重各半。`attended_xy` 和当前位置 `state` 拼成 `[B, 4]`，经过 `Linear → ReLU → Linear → Tanh`，再整理成 `[B, 4, 2]`。`Tanh` 的结果乘以 `MAX_STEP`，得到受限的每轴位移。

`lesson.py` 的 `train()` 用 `DataLoader` 提供批次，`MSELoss` 比较预测动作与专家动作，`loss.backward()` 自动把梯度沿动作头和注意力传回 Query，最后用 `AdamW` 更新权重。`--steps` 指优化器更新次数，训练集沿用有放回抽样。指令与颜色的对应来自示教的动作监督。想追踪一次反传，可以在 `loss.backward()` 后观察 `model.queries.grad`。

这个视觉编码器利用了合成环境的先验：黑色是背景，目标各占一个像素，像素坐标和运动坐标共用同一平面。它展示了条件注意力的数据流。摄像机图像中的遮挡、投影、多视角和开放词汇需要另行建模。

## 第三课：离线误差与闭环效果

读 [`lesson.py`](lesson.py) 的 `rollout()`。离线 MSE 给模型的是专家走到的位置。闭环 rollout 则从同一个起点出发，每次执行预测块的前两步，然后使用**模型自己走到的位置**再次预测。评估在八步后测量离目标的欧氏距离，距离不超过 `0.12` 记为成功。

评估在 512 条独立测试回合上报告完整输入、交换指令、打乱图像和位置置零的结果。交换指令时仍按原目标评分，可以直接看语言改变后会不会走向另一处。

本机 CPU 按默认参数训练 1,200 步，验证 MSE 从 `0.021974` 降到 `0.000081`。测试到达率分别为完整输入 `100.0%`、交换指令 `0.0%`、打乱图像 `1.0%`、位置置零 `12.1%`。

## 动手实验

1. 用 `python lesson.py --checkpoint runs/short.pt train --steps 20` 训练一次，比较验证 MSE、`attended_xy` 和动作方向。评估这份权重时将同一个 `--checkpoint` 放在 `evaluate` 前。
2. 在 `rollout()` 中给第一次 `step()` 后的模拟位置加一个固定偏移。保持相同扰动，比较 `--execute 1` 和 `--execute 4`。记录哪一次先用新的状态重新规划，以及终点距离如何变化。
3. 把一个 episode 的标签故意错开一拍，重新训练并比较 MSE 和闭环。先用 `inspect` 确认可见的第一条标签具体错在何处。这是对时序对齐的直接检查。

## 回到实验室 PI0.5

先在实验室代码里找 **观察、动作标签、执行块** 三个边界。`models/pi05/so101_bimanual.py` 的 `SO101BimanualInputs` 把双臂样本映射成 OpenPI observation，`SO101BimanualOutputs` 把动作从填充维度取回 12 个关节目标。继而看 `experiments/vla/openpi_so101_bimanual_config.py` 的训练配置，再到 `deploy/pi/so101_runtime_policy.py` 查动作块何时被请求和执行。

| 概念 | 教学项目 | 实验室仓库的经典基线 |
| --- | --- | --- |
| 视觉与语言 | 单张 RGB、两条固定指令、像素注意力 | 三路相机输入槽、任务文本、OpenPI 观测适配 |
| 状态与动作 | 二维位置、二维位移 | SO101 双臂 12D，关节与夹爪按各自合同转换 |
| 训练与执行 | 动作回归 MSE，预测 4 步，默认执行前 2 步 | OpenPI PI0.5 训练，经典模型视图为 15 Hz / H15，执行由对应 release 和 Pi 运行时合同决定 |

