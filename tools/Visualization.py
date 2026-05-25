from PIL import Image, ImageDraw

# 创建 512x512 白底图像
image = Image.new("RGB", (512, 512), (255, 255, 255))
draw = ImageDraw.Draw(image)

# polygon 数据（枯藤老树昏鸦）
polygon = [(30, 100), (90, 100), (90, 470), (30, 470)]

# 画出红色框
draw.polygon(polygon, outline="red", width=2)

# 显示或保存
image.show()
image.save("output_box_only.png")
