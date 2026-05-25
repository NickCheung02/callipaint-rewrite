#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# for Chinese:
python eval/clipscore.py \
        --img_dir /data/vdb/yuxiang.tyx/AIGC/eval/anytext2_wukong_generated \
        --input_json /data/vdb/yuxiang.tyx/AIGC/data/wukong_word/test1k.json

# for English:  change img_dir to .../anytext2_laion_generated and input_json to .../laion_word/test1k.json
# for long caption evaluation:  change .../test1k.json to .../test1k_long.json