
import torch
import open_clip

# 1. 选择设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print("使用设备:", device)

# 2. 加载预训练 CLIP
print("正在加载 OpenCLIP 模型...")

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k",
    device=device
)

# 3. 加载对应的文本 Tokenizer
tokenizer = open_clip.get_tokenizer("ViT-B-32")

model.eval()

print("模型加载成功！")
print("模型参数量:", sum(p.numel() for p in model.parameters()))
