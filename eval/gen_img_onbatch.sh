#!/bin/bash

# =============================================
# 自动运行 AnyText2 评估脚本 —— 批量模式
# 对指定目录下所有 .ckpt 文件逐一运行评估
# 输出目录：日期 + 自动递增版本号 + ckpt文件名（如 20250923V1-epoch29-step3360）
# 配置记录：每个运行保存为 config.txt
# =============================================

# ========== 基础路径配置 ==========
BASE_OUTPUT_DIR="./test-result"  # 基础输出目录
CKPT_DIR="/home/610-zzy/AnyText2-main-Real0922-DoubleStage-FHS-4-Calli/checkpoints/lightning_logs/version_2/checkpoints"
INPUT_JSON="test-result/poem_info2.json"  # 输入 JSON 文件路径

# ========== 获取所有 .ckpt 文件 ==========
CKPT_FILES=("$CKPT_DIR"/*.ckpt)

if [ ${#CKPT_FILES[@]} -eq 0 ] || [ ! -f "${CKPT_FILES[0]}" ]; then
    echo "❌ 未找到任何 .ckpt 文件在目录: $CKPT_DIR"
    exit 1
fi

echo "🔍 找到 ${#CKPT_FILES[@]} 个 .ckpt 文件，即将逐一运行评估..."

# ========== 公共日期前缀 ==========
DATE=$(date +%Y%m%d)  # 格式：20250923

# ========== 初始化全局计数器 ==========
GLOBAL_COUNTER=1

# ========== 成功/失败统计 ==========
SUCCESS_COUNT=0
FAILURE_COUNT=0

# ========== 遍历每个 ckpt 文件 ==========
for CKPT_PATH in "${CKPT_FILES[@]}"; do
    if [ ! -f "$CKPT_PATH" ]; then
        echo "⚠️  跳过无效文件: $CKPT_PATH"
        continue
    fi

    # 提取 ckpt 文件名（不含路径和扩展名），用于子目录命名
    CKPT_NAME=$(basename "$CKPT_PATH" .ckpt)

    # 自动生成唯一输出子目录（避免冲突）
    while [[ -d "${BASE_OUTPUT_DIR}/${DATE}V${GLOBAL_COUNTER}-${CKPT_NAME}" ]]; do
        ((GLOBAL_COUNTER++))
    done
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${DATE}V${GLOBAL_COUNTER}-${CKPT_NAME}"
    ((GLOBAL_COUNTER++))

    # 创建目录
    mkdir -p "$OUTPUT_DIR"
    echo "📁 输出目录: $OUTPUT_DIR (对应 $CKPT_NAME.ckpt)"

    # ========== 生成 config.txt 记录参数 ==========
    CONFIG_FILE="$OUTPUT_DIR/config.txt"

    {
        echo "========================================"
        echo "      AnyText2 评估配置记录"
        echo "========================================"
        echo "运行时间: $(date '+%Y年%m月%d日 %H:%M:%S')"
        echo "输出路径: $OUTPUT_DIR"
        echo "Checkpoint: $CKPT_PATH"
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
    echo "🚀 正在启动评估 ($CKPT_NAME.ckpt) ..."
    # 这一行会调用我们修改后的 Python 脚本
    python eval/anytext2_singleGPU.py \
        --ckpt_path "$CKPT_PATH" \
        --input_json "$INPUT_JSON" \
        --output_dir "$OUTPUT_DIR"

    # ========== 执行结果反馈 ==========
    if [ $? -eq 0 ]; then
        echo "🎉 评估成功！结果保存在: $OUTPUT_DIR"
        ((SUCCESS_COUNT++))
    else
        echo "❌ 评估失败，请检查错误。"
        ((FAILURE_COUNT++))
    fi

    echo "────────────────────────────────────────"
done

# ========== 最终汇总 ==========
echo "📊 评估完成！"
echo "✅ 成功: $SUCCESS_COUNT"
echo "❌ 失败: $FAILURE_COUNT"
echo "📁 所有结果保存在: $BASE_OUTPUT_DIR"

if [ $FAILURE_COUNT -gt 0 ]; then
    echo "⚠️  有失败任务，请检查对应目录下的日志。"
    exit 1
else
    echo "🎉 所有评估任务均成功完成！"
fi
# ========== 结束 ==========