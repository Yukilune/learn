"""二维环境、专家示教和从 episode 切出来的监督样本。"""

from dataclasses import dataclass

import numpy as np


IMAGE_SIZE = 8
HORIZON = 4
EPISODE_STEPS = 8
MAX_STEP = 0.35
INSTRUCTIONS = ("move to the red target", "move to the blue target")
LAYOUT_SPLITS = {
    "train": slice(0, 1000),
    "validation": slice(1000, 1150),
    "test": slice(1150, 1406),
    "demo": slice(1406, 1407),
}

# 图像行向下、列向右。坐标和动作统一按 (x, y) 排列。
axis = np.linspace(-1, 1, IMAGE_SIZE, dtype=np.float32)
PIXEL_XY = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)


@dataclass
class Scenes:
    image: np.ndarray        # [B, 8, 8, 3]，RGB
    instruction: np.ndarray  # [B]，固定句子的 token ID
    start: np.ndarray        # [B, 2]
    goal: np.ndarray         # [B, 2]，仅专家和评估读取


def tokenize(text: str) -> int:
    """这个固定词表用一个 token 表示整句指令。"""
    return INSTRUCTIONS.index(" ".join(text.lower().split()))


def make_scenes(split: str) -> Scenes:
    """先按布局划分，再为每个布局生成红、蓝两种指令。"""
    first, second = np.triu_indices(len(PIXEL_XY), k=1)
    separated = np.linalg.norm(PIXEL_XY[first] - PIXEL_XY[second], axis=1) >= 0.7
    targets = np.stack((first[separated], second[separated]), axis=1)
    rng = np.random.default_rng(101)
    rng.shuffle(targets)
    # 上三角枚举让同一对位置只出现一次，交换颜色后也不会跨集合。
    swap_colors = rng.random(len(targets)) < 0.5
    targets[swap_colors] = targets[swap_colors, ::-1]
    starts = rng.uniform(-1, 1, size=(len(targets), 2)).astype(np.float32)
    layout_slice = LAYOUT_SPLITS[split]
    targets = targets[layout_slice]
    count = len(targets)
    image = np.zeros((count, IMAGE_SIZE**2, 3), dtype=np.float32)
    image[np.arange(count), targets[:, 0], 0] = 1
    image[np.arange(count), targets[:, 1], 2] = 1
    tokens = np.array([tokenize(text) for text in INSTRUCTIONS])
    return Scenes(
        image=np.repeat(image.reshape(count, IMAGE_SIZE, IMAGE_SIZE, 3), 2, axis=0),
        instruction=np.tile(tokens, count),
        start=np.repeat(starts[layout_slice], 2, axis=0),
        goal=PIXEL_XY[targets.reshape(-1)],
    )


def step(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    return np.clip(state + np.clip(action, -MAX_STEP, MAX_STEP), -1, 1)


def expert_action(state: np.ndarray, goal: np.ndarray) -> np.ndarray:
    return np.clip(0.5 * (goal - state), -MAX_STEP, MAX_STEP)


def expert_episodes(scenes: Scenes) -> tuple[np.ndarray, np.ndarray]:
    """返回 episode 的状态 [B,T+1,2] 和发生在相应状态后的动作 [B,T,2]。"""
    states = [scenes.start]
    actions = []
    for _ in range(EPISODE_STEPS):
        action = expert_action(states[-1], scenes.goal)
        actions.append(action)
        states.append(step(states[-1], action))
    return np.stack(states, axis=1), np.stack(actions, axis=1)


def training_windows(scenes: Scenes) -> tuple[np.ndarray, ...]:
    """每个 observation_t 对齐 action[t:t+H]，只取能凑齐 H 步的窗口。"""
    states, actions = expert_episodes(scenes)
    windows = EPISODE_STEPS - HORIZON + 1
    chunks = np.stack([actions[:, t:t + HORIZON] for t in range(windows)], axis=1)
    return (
        np.repeat(scenes.image, windows, axis=0),
        np.repeat(scenes.instruction, windows),
        states[:, :windows].reshape(-1, 2),
        chunks.reshape(-1, HORIZON, 2),
    )
