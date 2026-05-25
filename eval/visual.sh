#!/bin/bash

# 可视化多边形标注的批处理脚本
# 使用方法: ./visualize_polygons.sh

# 设置变量
JSON_PATH="/home/610-zzy/AnyText2-main-Real724/test-result/test_data.json"          # 替换为你的JSON文件路径
OUTPUT_DIR="test-result/datavisual"         # 替换为你想保存图片的目录

# 检查输入文件是否存在
if [ ! -f "$JSON_PATH" ]; then
    echo "错误: JSON文件不存在: $JSON_PATH"
    echo "请编辑脚本并正确设置 JSON_PATH"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "开始处理..."
echo "JSON文件: $JSON_PATH"
echo "输出目录: $OUTPUT_DIR"

# 运行Python脚本
python eval/Visualization.py --json_path "$JSON_PATH" --output_dir "$OUTPUT_DIR"

# 检查执行是否成功
if [ $? -eq 0 ]; then
    echo "✅ 可视化完成！图片已保存至: $OUTPUT_DIR"
else
    echo "❌ 执行失败，请检查错误信息。"
    exit 1
fi