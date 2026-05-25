import os
import argparse
import numpy as np
import cv2
import time  # 引入时间模块，用于计算耗时
from PIL import Image
from ms_wrapper import AnyText2Model
from util import save_images, check_channels, resize_image

# --- 默认参数，可根据需要修改 ---
DEFAULT_A_PROMPT = 'best quality, extremely detailed, 4k, HD, super legible text, clear text edges, clear strokes, neat writing, no watermarks'
DEFAULT_N_PROMPT = 'low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture'

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Command-line inference for AnyText2.")
    
    # --- 核心输入 ---
    parser.add_argument('--img_prompt', type=str, required=True, help='Image prompt describing the scene.')
    parser.add_argument('--text_prompt', type=str, required=True, help='Text to be written, with each line in double quotes. E.g., "\\"Hello\\" \\"World\\""')
    parser.add_argument('--model_path', type=str, default='./models/anytext_v2.0.ckpt', help='Path to the AnyText2 checkpoint file.')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save the generated images.')
    
    # --- 位置图 (可选) ---
    parser.add_argument('--pos_img_path', type=str, default=None, help='[Optional] Path to a position image. If not provided, a blank image will be used.')

    # --- 字体和颜色 (简化) ---
    parser.add_argument('--font_path', type=str, default='font/Arial_Unicode.ttf', help='Path to the .ttf or .otf font file.')
    # 颜色格式为 "R,G,B"，例如 "255,0,0" 代表红色。500,500,500为随机颜色。
    parser.add_argument('--text_color', type=str, default='500,500,500', help='Comma-separated RGB values for the text color, e.g., "255,0,0". Use "500,500,500" for random colors.')

    # --- 推理参数 ---
    parser.add_argument('--seed', type=int, default=-1, help='Seed for random generation. -1 for random.')
    parser.add_argument('--img_count', type=int, default=1, help='Number of images to generate.')
    parser.add_argument('--ddim_steps', type=int, default=20, help='Number of DDIM steps.')
    parser.add_argument('--image_width', type=int, default=512, help='Width of the generated image.')
    parser.add_argument('--image_height', type=int, default=512, help='Height of the generated image.')
    parser.add_argument('--strength', type=float, default=1.0, help='Control strength.')
    parser.add_argument('--cfg_scale', type=float, default=7.5, help='CFG scale.')
    parser.add_argument('--a_prompt', type=str, default=DEFAULT_A_PROMPT, help='Added prompt.')
    parser.add_argument('--n_prompt', type=str, default=DEFAULT_N_PROMPT, help='Negative prompt.')
    
    return parser.parse_args()

def main():
    """主执行函数"""
    args = parse_arguments()
    
    # 1. 加载模型
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- 脚本启动 ---")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 正在加载模型: {args.model_path}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 这是一个耗时操作，可能需要几分钟，请耐心等待...")
    
    start_time = time.time()
    infer_params = {
        'use_fp16': True,
        'model_path': args.model_path
    }
    inference_model = AnyText2Model(model_dir='./models', **infer_params).cuda(0)
    end_time = time.time()
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- 模型加载成功！耗时: {end_time - start_time:.2f} 秒 ---")
    
    # 2. 准备输入数据
    # ... (这部分代码很快，无需修改) ...
    if args.pos_img_path and os.path.exists(args.pos_img_path):
        pos_imgs = cv2.imread(args.pos_img_path)
        pos_imgs = cv2.cvtColor(pos_imgs, cv2.COLOR_BGR2RGB)
    else:
        pos_imgs = np.zeros((args.image_height, args.image_width, 3), dtype=np.uint8)

    params = {
        "mode": "gen", "sort_priority": "↕", "revise_pos": False,
        "image_count": args.img_count, "ddim_steps": args.ddim_steps,
        "image_width": args.image_width, "image_height": args.image_height,
        "strength": args.strength, "cfg_scale": args.cfg_scale, "eta": 0.0,
        "a_prompt": args.a_prompt, "n_prompt": args.n_prompt, "show_debug": False,
        "glyline_font_path": [args.font_path], "text_colors": args.text_color
    }
    input_data = {
        "img_prompt": args.img_prompt, "text_prompt": args.text_prompt,
        "seed": args.seed, "draw_pos": pos_imgs, "ori_image": None
    }
    
    # 3. 执行推理
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- 输入准备就绪，即将开始推理... ---")
    start_time = time.time()
    results, rtn_code, rtn_warning, debug_info = inference_model(input_data, **params)
    end_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- 推理完成！耗时: {end_time - start_time:.2f} 秒 ---")
    
    # 4. 保存结果
    if rtn_code >= 0:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        save_images(results, args.output_dir)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 结果已成功保存至: {args.output_dir}")
        if rtn_warning:
            print(f"警告: {rtn_warning}")
    else:
        print(f"推理出错: {rtn_warning}")

if __name__ == '__main__':
    main()