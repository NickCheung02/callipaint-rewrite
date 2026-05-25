from PIL import Image

# 图片文件的路径
image_path = "test-result/20250730v3/modern_shanshui_001_1.jpg"  # <-- 替换成你的图片路径

try:
    # 打开图片文件
    with Image.open(image_path) as img:
        # 获取图片的尺寸
        width, height = img.size
        
        # 打印尺寸
        print(f"图片路径: {image_path}")
        print(f"图片尺寸 (宽 x 高): {width} x {height} 像素")
        print(f"宽度: {width} 像素")
        print(f"高度: {height} 像素")

except FileNotFoundError:
    print(f"错误：找不到文件 '{image_path}'")
except Exception as e:
    print(f"发生错误: {e}")