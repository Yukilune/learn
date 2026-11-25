"""VLA 教学入口：inspect → train → evaluate → demo。"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from data import EPISODE_STEPS, HORIZON, INSTRUCTIONS, Scenes
from data import expert_episodes, make_scenes, step, tokenize, training_windows
from model import TinyVLA


CHECKPOINT = Path(__file__).parent / "runs" / "tiny_vla.pt"
SUCCESS_RADIUS = 0.12
BATCH_SIZE = 128


def observation_tensors(
    image: np.ndarray, tokens: np.ndarray, state: np.ndarray,
) -> tuple[Tensor, Tensor, Tensor]:
    return torch.from_numpy(image), torch.as_tensor(tokens, dtype=torch.long), torch.from_numpy(state)


def window_dataset(scenes: Scenes) -> TensorDataset:
    image, tokens, state, chunks = training_windows(scenes)
    return TensorDataset(*observation_tensors(image, tokens, state), torch.from_numpy(chunks))


@torch.inference_mode()
def rollout(model: TinyVLA, scenes: Scenes, execute: int, ablation: str = "full"):
    """每次重新看图和当前状态，执行动作块的前 execute 步。"""
    image = scenes.image
    tokens = scenes.instruction
    if ablation == "swap_instruction":
        tokens = tokens ^ 1
    if ablation == "shuffle_image":
        image = np.roll(image, 2, axis=0)  # 跨布局，保持颜色分布与输入形状
    state = scenes.start.copy()
    image_tensor, token_tensor, _ = observation_tensors(image, tokens, state)
    trajectory = [state]
    for t in range(0, EPISODE_STEPS, execute):
        observed_state = np.zeros_like(state) if ablation == "zero_state" else state
        chunk, _ = model(image_tensor, token_tensor, torch.from_numpy(observed_state))
        for action in chunk[:, :min(execute, EPISODE_STEPS - t)].numpy().transpose(1, 0, 2):
            state = step(state, action)
            trajectory.append(state)
    return np.stack(trajectory, axis=1)


def metrics(final: np.ndarray, goal: np.ndarray) -> tuple[float, float]:
    distance = np.linalg.norm(final - goal, axis=-1)
    return float(distance.mean()), float(np.mean(distance <= SUCCESS_RADIUS))


@torch.inference_mode()
def validation_loss(model: TinyVLA, dataset: TensorDataset, criterion: nn.MSELoss) -> float:
    image, tokens, state, expert_chunk = dataset.tensors
    predicted, _ = model(image, tokens, state)
    return criterion(predicted, expert_chunk).item()


def train(steps: int, checkpoint: Path) -> None:
    train_set = window_dataset(make_scenes("train"))
    validation = window_dataset(make_scenes("validation"))
    torch.manual_seed(0)
    # CLI 按更新步数训练。保留原先有放回抽样的语义。
    sampler = RandomSampler(
        train_set,
        replacement=True,
        num_samples=steps * BATCH_SIZE,
        generator=torch.Generator().manual_seed(0),
    )
    batches = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler)
    model = TinyVLA()
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.012)
    print(f"windows: train={len(train_set)}, validation={len(validation)}")
    model.eval()
    print(f"step=0 validation_mse={validation_loss(model, validation, criterion):.6f}")
    model.train()
    for iteration, (image, tokens, state, expert_chunk) in enumerate(batches, start=1):
        optimizer.zero_grad(set_to_none=True)
        predicted, _ = model(image, tokens, state)
        loss = criterion(predicted, expert_chunk)
        loss.backward()
        optimizer.step()
        if iteration % 200 == 0 or iteration == steps:
            model.eval()
            print(f"step={iteration} batch_mse={loss.item():.6f} "
                  f"validation_mse={validation_loss(model, validation, criterion):.6f}")
            model.train()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    print(f"saved {checkpoint}")


def inspect() -> None:
    scenes = make_scenes("demo")
    images, tokens, states, chunks = training_windows(scenes)
    print(f"image={images.shape} tokens={tokens.shape} state={states.shape} actions={chunks.shape}")
    print("Same image & same start, two instructions:")
    np.set_printoptions(precision=3, suppress=True)
    for token in range(len(INSTRUCTIONS)):
        index = token * (EPISODE_STEPS - HORIZON + 1)
        print(f"{INSTRUCTIONS[token]} | state={states[index]} | expert goal={scenes.goal[token]}")
        print(chunks[index])


def evaluate(model: TinyVLA, execute: int) -> None:
    scenes = make_scenes("test")
    states, _ = expert_episodes(scenes)
    print(f"test episodes={len(scenes.start)}, predict={HORIZON}, execute={execute}, "
          f"success radius={SUCCESS_RADIUS}")
    print(f"test expert-window MSE: {validation_loss(model, window_dataset(scenes), nn.MSELoss()):.6f}")
    results = {"expert": states[:, -1]}
    for mode in ("full", "swap_instruction", "shuffle_image", "zero_state"):
        results[mode] = rollout(model, scenes, execute, mode)[:, -1]
    for name, final in results.items():
        distance, success = metrics(final, scenes.goal)
        print(f"{name:>16}  final_distance={distance:.4f}  success={success:.1%}")


@torch.inference_mode()
def demo(model: TinyVLA, instruction: str, execute: int) -> None:
    token = tokenize(instruction)
    pair = make_scenes("demo")
    index = slice(token, token + 1)
    scenes = Scenes(pair.image[index], pair.instruction[index], pair.start[index], pair.goal[index])
    chunk, attended = model(*observation_tensors(scenes.image, scenes.instruction, scenes.start))
    trajectory = rollout(model, scenes, execute)[0]
    np.set_printoptions(precision=3, suppress=True)
    print(f"{INSTRUCTIONS[token]} | attended_xy={attended[0].numpy()} | goal={scenes.goal[0]}")
    print(f"first predicted action chunk:\n{chunk[0].numpy()}")
    print(f"trajectory (start, then state after each action):\n{trajectory}")


def load_model(checkpoint: Path) -> TinyVLA:
    model = TinyVLA()
    if checkpoint.suffix.lower() == ".npz":
        with np.load(checkpoint, allow_pickle=False) as weights:
            # 旧权重是 [in, out]，nn.Linear 权重是 [out, in]。
            parameters = {
                "queries": torch.from_numpy(weights["queries"]),
                "action_head.0.weight": torch.from_numpy(weights["input_weight"].T),
                "action_head.0.bias": torch.from_numpy(weights["input_bias"]),
                "action_head.2.weight": torch.from_numpy(weights["output_weight"].T),
                "action_head.2.bias": torch.from_numpy(weights["output_bias"]),
            }
    else:
        parameters = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for name, parameter in parameters.items():
        if not torch.isfinite(parameter).all():
            raise ValueError(f"Invalid checkpoint tensor: {name}")
    model.load_state_dict(parameters)
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect")
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--steps", type=int, default=1200)
    evaluate_parser = commands.add_parser("evaluate")
    demo_parser = commands.add_parser("demo")
    demo_parser.add_argument("--instruction", choices=INSTRUCTIONS, default=INSTRUCTIONS[0])
    for command in (evaluate_parser, demo_parser):
        command.add_argument("--execute", type=int, choices=range(1, HORIZON + 1), default=2)
    args = parser.parse_args()
    # 小批量、小模型在 CPU 上用单线程，避免并行调度占主要开销。
    torch.set_num_threads(1)
    if args.command == "inspect":
        inspect()
    elif args.command == "train":
        if args.steps < 1:
            parser.error("--steps must be positive")
        if args.checkpoint.suffix.lower() == ".npz":
            parser.error(".npz checkpoints are read-only; choose a .pt path for training")
        train(args.steps, args.checkpoint)
    else:
        if not args.checkpoint.is_file():
            parser.error(f"checkpoint unavailable: {args.checkpoint}. Run the train command first.")
        model = load_model(args.checkpoint)
        if args.command == "evaluate":
            evaluate(model, args.execute)
        else:
            demo(model, args.instruction, args.execute)


if __name__ == "__main__":
    main()
