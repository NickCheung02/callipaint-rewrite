import os
import json
from PIL import Image, ImageDraw
import numpy as np

# 设置路径
json_path = 'poem_data/new_data.json'       # 替换为你的 JSON 文件路径
image_root = 'data/poem_data/imgs_0'  # 替换为你存放图片的根目录
output_dir = 'data/new_calli'  # 输出抠图的目录

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 读取 JSON 文件
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 黑色阈值
BLACK_THRESHOLD = 500  # 可以根据实际情况调整

# 处理每一张图
for item in data['data_list']:
    img_name = item['img_name']
    img_path = os.path.join(image_root, img_name)
    if not os.path.exists(img_path):
        print(f"图片不存在: {img_path}")
        continue

    # 打开图像
    image = Image.open(img_path).convert("RGB")
    img_array = np.array(image)

    # 遍历 annotations 中的每个 polygon 区域
    for idx, anno in enumerate(item['annotations']):
        polygon = anno['polygon']
        text = anno['text']

        # 构建 mask
        mask = Image.new('L', image.size, 0)
        draw = ImageDraw.Draw(mask)
        polygon_tuple = [(int(x), int(y)) for x, y in polygon]
        draw.polygon(polygon_tuple, outline=1, fill=1)
        mask_np = np.array(mask)

        # 提取多边形区域
        masked_img = np.zeros_like(img_array)
        masked_img[mask_np == 1] = img_array[mask_np == 1]

        # 获取边界框并裁剪
        xs, ys = zip(*polygon_tuple)
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        cropped = masked_img[y_min:y_max+1, x_min:x_max+1]
        mask_cropped = mask_np[y_min:y_max+1, x_min:x_max+1]

        # 创建RGB图像，背景为白色
        result_img = np.ones((cropped.shape[0], cropped.shape[1], 3), dtype=np.uint8) * 255
        
        # 计算RGB总和，判断是否为黑色
        gray_sum = np.sum(cropped[:, :, :3], axis=2)
        black_mask = (gray_sum < BLACK_THRESHOLD) & (mask_cropped == 1)
        
        # 只保留黑色部分，背景为白色
        result_img[black_mask] = cropped[black_mask]

        # 转换回 PIL 图像
        pil_result = Image.fromarray(result_img, 'RGB')
        
        # 转换为灰度图
        gray_img = pil_result.convert('L')
        
        # 二值化处理 - 小于128的变为黑色(0)，大于等于128的变为白色(255)
        binary_img = gray_img.point(lambda x: 0 if x < 128 else 255, '1')

        # 保存文件
        output_filename = f"{os.path.splitext(img_name)[0]}_calligraphy_{idx}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        binary_img.save(output_path, 'JPEG', quality=95)

        print(f"已保存二值化书法图像: {output_path}")