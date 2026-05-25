#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# for Chinese
python eval/eval_dgocr.py \
        --img_dir test-result/20250730v1 \
        --input_json test-result/processed_data.json
# for English:  change img_dir to .../anytext2_laion_generated and input_json to .../laion_word/test1k.json
# for long caption evaluation:  change .../test1k.json to .../test1k_long.json
