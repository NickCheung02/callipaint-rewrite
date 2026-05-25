#!/bin/bash

# --- 在脚本内部初始化并激活 Conda ---
conda activate zzy-anytext

echo "--- conda环境 'zzy-anytext' 已激活, 准备执行推理脚本... ---"

# --- 您提供的、完整的 Python 命令 ---
python inference.py \
  --img_prompt "A cute cat holding a sign, cartoon style" \
  --text_prompt "\"你好\" \"世界\"" \
  --model_path "checkpoints/lightning_logs/version_0/checkpoints/epoch=29-step=2760.ckpt" \
  --output_dir "test/output"

echo "--- Python 脚本执行完毕 ---"