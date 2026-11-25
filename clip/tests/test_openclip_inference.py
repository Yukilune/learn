
import torch
import torch.nn.functional as F
import open_clip
import pyarrow.parquet as pq

from pathlib import Path
from io import BytesIO
from PIL import Image


def main():
    # 1. 选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("使用设备:", device)

    # 2. 加载 OpenCLIP
    print("正在加载模型...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device=device
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    print("模型加载成功！")

    # 3. 读取 Flickr30k 数据
    parquet_path = Path(
        r"F:\datasets\flickr30k\TEST\test-00000-of-00009.parquet"
    )

    parquet_file = pq.ParquetFile(parquet_path)

    table = parquet_file.read_row_group(
        0,
        columns=["image", "caption"]
    )

    row = table.slice(0, 1).to_pylist()[0]

    # 4. 读取图片
    image_bytes = row["image"]["bytes"]

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    # 使用 OpenCLIP 自己的预处理流程
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # 5. 读取五条英文描述
    captions = row["caption"]

    text_tokens = tokenizer(captions).to(device)

    print("\n图片 Tensor Shape:", image_tensor.shape)
    print("文本 Token Shape:", text_tokens.shape)

    # 6. 提取图像和文本特征
    with torch.inference_mode():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)

        # L2 归一化
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # 计算余弦相似度
        similarities = image_features @ text_features.T

    # 7. 输出结果
    print("\n图文匹配结果：")

    scores = similarities[0].cpu().tolist()

    results = sorted(
        zip(captions, scores),
        key=lambda x: x[1],
        reverse=True
    )

    for rank, (caption, score) in enumerate(results, 1):
        print(f"\nTop {rank}")
        print(f"相似度: {score:.4f}")
        print(f"描述: {caption}")


if __name__ == "__main__":
    main()
