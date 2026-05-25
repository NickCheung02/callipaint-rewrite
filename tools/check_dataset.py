import os
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

# 配置路径
json_path = "poem_data/poem_data.json"  # JSON文件路径
save_dir = "data/datacheck/3"            # 输出目录
os.makedirs(save_dir, exist_ok=True)

# 字体路径（支持中文）
font_path = "font/lang_font/钉钉进步体.ttf"
font_size = 20
font = ImageFont.truetype(font_path, font_size)

# 换行函数（按像素宽度）
def wrap_text(text, draw, font, max_width):
    lines = []
    for paragraph in text.split("\n"):
        line = ''
        for char in paragraph:
            if draw.textlength(line + char, font=font) <= max_width:
                line += char
            else:
                lines.append(line)
                line = char
        if line:
            lines.append(line)
    return lines

# 读取 JSON 数据
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 遍历数据列表
for item in data["data_list"]:
    img_name = item["img_name"]
    img_path = os.path.join(data["data_root"], img_name)

    # 读取图像
    image = cv2.imread(img_path)
    if image is None:
        print(f"❌ 图片读取失败：{img_path}")
        continue

    # 绘制所有 polygon 多边形
    for anno in item["annotations"]:
        pts = np.array(anno["polygon"], np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # 准备文本内容
    text_info = f"Image Name: {img_name}\n\n画面提示词: {item['caption']}\n\n文本内容: {item['annotations'][0]['text']}"
    h, w, _ = image.shape
    info_width = 400
    padding = 20

    # 临时 image 画布用于计算文本高度
    dummy_img = Image.new("RGB", (info_width, 1000), "white")
    draw = ImageDraw.Draw(dummy_img)
    lines = wrap_text(text_info, draw, font, info_width - 2 * padding)
    line_height = font.getsize("测")[1] + 6
    text_height = line_height * len(lines) + 2 * padding

    # 创建最终 info 区域图像
    info_img = Image.new("RGB", (info_width, max(h, text_height)), "white")
    draw = ImageDraw.Draw(info_img)
    for i, line in enumerate(lines):
        draw.text((padding, padding + i * line_height), line, font=font, fill=(0, 0, 0))

    # 拼接图像（宽度 = 原图宽 + 文字宽；高度 = max(原图高, 文本高)）
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if h < text_height:
        image_pil = ImageOps.pad(image_pil, (w, text_height), color="white")

    combined = Image.new("RGB", (w + info_width, max(h, text_height)), "white")
    combined.paste(image_pil, (0, 0))
    combined.paste(info_img, (w, 0))

    # 保存
    save_path = os.path.join(save_dir, img_name)
    combined.save(save_path)
    print(f"✅ 已保存: {save_path}")
