
import os
import csv
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# =========================
# 1. 路径与基本设置
# =========================

# 测试集文件夹
IMAGE_DIR = r"F:\数据集\测试集\test_set\test_set\dc"

# 模型文件：默认与本脚本放在同一目录
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "best_catdog_model.pth"
)

IMG_SIZE = 64
EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("当前设备：", device)
print("测试集目录：", IMAGE_DIR)
print("模型文件：", MODEL_PATH)


# =========================
# 2. 定义与训练时相同的 CNN
# =========================

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================
# 3. 图片预处理
# 必须与验证集预处理保持一致
# =========================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


# =========================
# 4. 加载最优模型
# =========================

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"找不到模型文件：{MODEL_PATH}\n"
        "请确认 best_catdog_model.pth 与预测脚本在同一目录，"
        "或者修改 MODEL_PATH。"
    )

model = SimpleCNN().to(device)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()

print("最优模型加载成功！")


# =========================
# 5. 查找测试图片
# =========================

if not os.path.isdir(IMAGE_DIR):
    raise FileNotFoundError(
        f"测试集文件夹不存在：{IMAGE_DIR}"
    )

image_names = sorted([
    name for name in os.listdir(IMAGE_DIR)
    if name.lower().endswith(EXTENSIONS)
    and os.path.isfile(os.path.join(IMAGE_DIR, name))
])

print("找到图片数量：", len(image_names))

if not image_names:
    raise RuntimeError(
        "没有找到图片，请检查路径和图片格式。"
    )


# =========================
# 6. 逐张预测并评估
# =========================

total = 0
correct = 0
wrong_images = []
results = []

with torch.no_grad():
    for name in image_names:
        path = os.path.join(IMAGE_DIR, name)

        try:
            # 读取图片并转换为 RGB
            with Image.open(path) as img:
                img = img.convert("RGB")
                image = transform(img)

            # [3, 64, 64] -> [1, 3, 64, 64]
            image = image.unsqueeze(0).to(device)

            # 模型输出两个类别的原始分数
            outputs = model(image)

            # 转换成类别概率
            probabilities = torch.softmax(outputs, dim=1)

            # 取概率最大的类别和置信度
            confidence, prediction = probabilities.max(dim=1)

            pred_id = prediction.item()
            label = "猫" if pred_id == 0 else "狗"
            score = confidence.item() * 100

            # 从文件名获取真实标签
            lower_name = name.lower()

            if lower_name.startswith("cat"):
                true_label = "猫"
            elif lower_name.startswith("dog"):
                true_label = "狗"
            else:
                true_label = None

            # 只有能确定真实标签时，才参与准确率统计
            if true_label is not None:
                total += 1

                if label == true_label:
                    correct += 1
                    result = "正确"
                else:
                    result = "错误"
                    wrong_images.append(
                        (name, true_label, label, score)
                    )

                print(
                    f"{name} | 真实：{true_label} | "
                    f"预测：{label} | "
                    f"置信度：{score:.2f}% | {result}"
                )
            else:
                result = "无真实标签"
                print(
                    f"{name} | 预测：{label} | "
                    f"置信度：{score:.2f}% | "
                    "无法从文件名确定真实标签"
                )

            results.append([
                name,
                true_label if true_label is not None else "",
                label,
                f"{score:.2f}%",
                result
            ])

        except Exception as e:
            print(f"{name} | 读取或预测失败：{e}")


# =========================
# 7. 输出整体评估结果
# =========================

print("\n========== 评估结果 ==========")
print(f"参与评估的图片：{total}")
print(f"预测正确：{correct}")
print(f"预测错误：{total - correct}")

if total > 0:
    accuracy = correct / total * 100
    print(f"整体准确率：{accuracy:.2f}%")
else:
    print("没有可用于计算准确率的图片。")

# print("\n========== 错误图片 ==========")

# if wrong_images:
#     for name, true_label, label, score in wrong_images:
#         print(
#             f"{name} | 真实：{true_label} | "
#             f"预测：{label} | 置信度：{score:.2f}%"
#         )
# else:
#     print("没有发现预测错误的图片。")


# =========================
# 8. 保存预测报告 CSV
# =========================

report_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "catdog_prediction_report.csv"
)

with open(
    report_path,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "图片名称",
        "真实标签",
        "预测标签",
        "置信度",
        "结果"
    ])
    writer.writerows(results)

print("\n预测报告已保存：", report_path)
print("全部预测完成！")
