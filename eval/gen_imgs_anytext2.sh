#!/bin/bash

# =============================================
# 自动运行 AnyText2 评估脚本
# 输出目录：日期 + 自动递增版本号（如 20250923V1）
# 配置记录：保存为 config.txt（普通文本文件）
# =============================================

# ========== 基础路径配置 ==========
BASE_OUTPUT_DIR="/home/610-zzy/AnyText2-main-Real0922-DoubleStage/test-result"  # 基础输出目录
CKPT_PATH="/home/610-zzy/AnyText2-main-Real0922-DoubleStage/checkpoints/lightning_logs/version_2/checkpoints/epoch=19-step=6000.ckpt"
INPUT_JSON="test-result/poem_info2.json"  # 输入 JSON 文件路径

# ========== 自动生成输出目录（日期 + 自动版本） ==========
DATE=$(date +%Y%m%d)  # 格式：20250923

# 自动检测并递增版本号 V1, V2, V3...
COUNTER=1
while [[ -d "${BASE_OUTPUT_DIR}/${DATE}V${COUNTER}" ]]; do
    ((COUNTER++))
done
VERSION="V${COUNTER}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${DATE}${VERSION}"

# 创建目录
mkdir -p "$OUTPUT_DIR"
echo "📁 输出目录: $OUTPUT_DIR"

# ========== 生成 config.txt 记录参数 ==========
CONFIG_FILE="$OUTPUT_DIR/config.txt"

{
    echo "========================================"
    echo "      AnyText2 评估配置记录"
    echo "========================================"
    echo "运行时间: $(date '+%Y年%m月%d日 %H:%M:%S')"
    echo "输出路径: $OUTPUT_DIR"
    echo ""
    echo "[执行命令]"
    echo "python eval/anytext2_singleGPU.py \\"
    echo "    --ckpt_path $CKPT_PATH \\"
    echo "    --input_json $INPUT_JSON \\"
    echo "    --output_dir $OUTPUT_DIR"
    echo ""
    echo "[系统信息]"
    echo "主机名: $(hostname)"
    echo "用户名: $(whoami)"
    if command -v git >/dev/null 2>&1 && [ -d .git ]; then
        COMMIT=$(git rev-parse HEAD 2>/dev/null)
        echo "Git提交: ${COMMIT:-获取失败}"
    else
        echo "Git提交: 未使用 Git 或仓库不存在"
    fi
} > "$CONFIG_FILE"

echo "✅ 配置已保存至: $CONFIG_FILE"

# ========== 执行评估脚本 ==========
echo "🚀 正在启动评估..."
python eval/anytext2_singleGPU.py \
    --ckpt_path "$CKPT_PATH" \
    --input_json "$INPUT_JSON" \
    --output_dir "$OUTPUT_DIR"

# ========== 执行结果反馈 ==========
if [ $? -eq 0 ]; then
    echo "🎉 评估成功！结果保存在: $OUTPUT_DIR"
else
    echo "❌ 评估失败，请检查错误。"
    exit 1
fi