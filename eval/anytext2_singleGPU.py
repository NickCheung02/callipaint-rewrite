import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import einops
import numpy as np
import torch
import random
from PIL import ImageFont

from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from t3_dataset import draw_glyph, draw_glyph2, get_text_caption
from dataset_util import load
from tqdm import tqdm
import argparse
import time

save_memory = False
# parameters
config_yaml = './models_yaml/anytext2_sd15.yaml'
ckpt_path = './models/anytext_v2.0.ckpt'
json_path = '/data/vdb/yuxiang.tyx/AIGC/data/laion_word/test1k-sample.json'
output_dir = '/home/610-zzy/AnyText2-main-Real0909-Calli/test-result/20250916v1'
num_samples = 10
image_resolution = 512
strength = 1.0
ddim_steps = 20
scale = 7.5
seed = 100
eta = 0.0
a_prompt = 'best quality, extremely detailed'
n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, watermark'
PLACE_HOLDER = '*'
max_chars = 20
max_lines = 20
font = ImageFont.truetype('font/FZQianLXSJW.TTF', size=60)
glyph_scale = 1
use_fp16 = True
default_color = [500, 500, 500]
fonthint_type = 'Arial'  # 'Arial' or 'None' or 'Hollow'
# --- NEW: 必须与 cldm.py 中的 context_prompt_keys 保持一致 ---
CONTEXT_PROMPT_KEYS = ['full_prompt', 'element_prompt', 'mood_prompt', 'style_prompt']


def parse_args():
    parser = argparse.ArgumentParser(description='generate images')
    parser.add_argument('--input_json', type=str, default=json_path)
    parser.add_argument('--output_dir', type=str, default=output_dir)
    parser.add_argument('--ckpt_path', type=str, default=ckpt_path)
    parser.add_argument('--config_yaml', type=str, default=config_yaml)
    args = parser.parse_args()
    return args


def arr2tensor(arr, bs):
    if len(arr.shape) == 3:
        arr = np.transpose(arr, (2, 0, 1))
    _arr = torch.from_numpy(arr.copy()).float().cuda()
    if use_fp16:
        _arr = _arr.half()
    _arr = torch.stack([_arr for _ in range(bs)], dim=0)
    return _arr


def load_data(input_path):
    content = load(input_path)
    d = []
    count = 0
    
    # --- FIX: 处理 content 可能是 list 或 dict 的情况 ---
    data_list = []
    if isinstance(content, dict) and 'data_list' in content:
        # 这是训练集格式 { "data_root": "...", "data_list": [...] }
        print("JSON format: Detected Dictionary with 'data_list' key (Training format)")
        data_list = content['data_list']
        # data_root = content.get('data_root', '.') # 如果需要拼接路径，可以在这里处理
    elif isinstance(content, list):
        # 这是推理集格式 [ { ... }, { ... } ]
        print("JSON format: Detected List (Inference format)")
        data_list = content  # content 本身就是数据列表
    else:
        print(f"Error: Unexpected JSON format in {input_path}. Expected a list or a dict with 'data_list' key.")
        return []
    # --- End FIX ---

    for gt in data_list: # <-- 修正：现在 data_list 肯定是一个列表
        info = {}
        info['img_name'] = gt['img_name']
        
        # --- MODIFIED: 加载所有提示词类型 ---
        default_caption = gt.get('caption', '')
        info['caption'] = default_caption  # 保留旧键
        info['full_prompt'] = gt.get('full_prompt', default_caption)
        info['element_prompt'] = gt.get('element_prompt', default_caption)
        info['mood_prompt'] = gt.get('mood_prompt', default_caption)
        info['style_prompt'] = gt.get('style_prompt', default_caption)
        # --- End MODIFIED ---

        if PLACE_HOLDER in info['caption']:
            count += 1
            # 清理所有可能包含占位符的提示词
            info['caption'] = info['caption'].replace(PLACE_HOLDER, " ")
            info['full_prompt'] = info['full_prompt'].replace(PLACE_HOLDER, " ")
            info['element_prompt'] = info['element_prompt'].replace(PLACE_HOLDER, " ")
            info['mood_prompt'] = info['mood_prompt'].replace(PLACE_HOLDER, " ")
            info['style_prompt'] = info['style_prompt'].replace(PLACE_HOLDER, " ")
            
        if 'annotations' in gt:
            polygons = []
            texts = []
            pos = []
            for annotation in gt['annotations']:
                if len(annotation['polygon']) == 0:
                    continue
                if 'valid' in annotation and annotation['valid'] is False:
                    continue
                polygons.append(annotation['polygon'])
                texts.append(annotation['text'])
                if 'pos' in annotation:
                    pos.append(annotation['pos'])
            if len(texts) == 0:  # in case empty text
                texts = [' ', ]
                polygons = [[[0, 0], [0, 50], [50, 50], [50, 0]], ]
                pos = [0, ]
            info['polygons'] = [np.array(i) for i in polygons]
            info['texts'] = texts
            info['pos'] = pos
        d.append(info)
    print(f'{input_path} loaded, imgs={len(d)}')
    if count > 0:
        print(f"Found {count} image's caption contain placeholder: {PLACE_HOLDER}, change to ' '...")
    return d


def draw_pos(ploygon, prob=1.0):
    img = np.zeros((512, 512, 1))
    if random.random() < prob:
        pts = ploygon.reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], color=255)
    return img/255.


def get_item(data_list, item):
    item_dict = {}
    cur_item = data_list[item]
    item_dict['img_name'] = cur_item['img_name']

    # --- MODIFIED: 传递所有提示词 ---
    item_dict['img_caption'] = cur_item['caption']  # 旧键
    item_dict['full_prompt'] = cur_item['full_prompt']
    item_dict['element_prompt'] = cur_item['element_prompt']
    item_dict['mood_prompt'] = cur_item['mood_prompt']
    item_dict['style_prompt'] = cur_item['style_prompt']
    # --- End MODIFIED ---

    item_dict['text_caption'] = ''
    item_dict['glyphs'] = []
    item_dict['gly_line'] = []
    item_dict['positions'] = []
    item_dict['texts'] = []
    item_dict['color'] = []
    texts = cur_item.get('texts', [])
    if len(texts) > 0:
        sel_idxs = [i for i in range(len(texts))]
        if len(texts) > max_lines:
            sel_idxs = sel_idxs[:max_lines]
        item_dict['text_caption'] = get_text_caption(len(sel_idxs), PLACE_HOLDER)
        item_dict['polygons'] = [cur_item['polygons'][i] for i in sel_idxs]
        item_dict['texts'] = [cur_item['texts'][i][:max_chars] for i in sel_idxs]
        item_dict['color'] += [np.array(default_color)] * len(sel_idxs)
        # glyphs
        for idx, text in enumerate(item_dict['texts']):
            gly_line = draw_glyph(font, text)
            glyphs = draw_glyph2(font, text, item_dict['polygons'][idx], item_dict['color'][idx], scale=glyph_scale)
            item_dict['glyphs'] += [glyphs]
            item_dict['gly_line'] += [gly_line]
        # mask_pos
        for polygon in item_dict['polygons']:
            item_dict['positions'] += [draw_pos(polygon, 1.0)]
    fill_caption = False
    if fill_caption:  # if using embedding_manager, DO NOT fill caption!
        for i in range(len(item_dict['texts'])):
            r_txt = item_dict['texts'][i]
            # 注意：这里只更新了旧的 'caption'，在多提示词模式下可能需要重新考虑
            item_dict['caption'] = item_dict['caption'].replace(PLACE_HOLDER, f'"{r_txt}"', 1)
    # padding
    n_lines = min(len(texts), max_lines)
    item_dict['n_lines'] = n_lines
    n_pad = max_lines - n_lines
    if n_pad > 0:
        item_dict['glyphs'] += [np.zeros((512*glyph_scale, 512*glyph_scale, 3))] * n_pad
        item_dict['gly_line'] += [np.zeros((80, 512, 1))] * n_pad
        item_dict['positions'] += [np.zeros((512, 512, 1))] * n_pad
        item_dict['color'] += [np.array(default_color)] * n_pad
    if fonthint_type == 'Arial':
        item_dict['font_hint'] = cv2.resize(np.sum(np.stack(item_dict['glyphs']), axis=0).clip(0, 1), (512, 512))  # use arial font as font_hint
    elif fonthint_type == 'None':
        item_dict['font_hint'] = np.zeros((512, 512, 3))  # don't specify font_hint
    elif fonthint_type == 'Hollow':
        font_hint_fg = cv2.resize(np.sum(np.stack(item_dict['glyphs']), axis=0).clip(0, 1), (512, 512))
        font_hint_fg = font_hint_fg[..., 0:1]*255
        if font_hint_fg.mean() > 0:
            img = cv2.imread('font/bg_noise.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (font_hint_fg.shape[1], font_hint_fg.shape[0]))
            img[img < 230] = 0
            font_hint_bg = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            kernel1 = np.ones((3, 3), dtype=np.uint8)
            kernel2 = np.ones((5, 5), dtype=np.uint8)
            dilate_img1 = cv2.dilate(font_hint_fg[..., 0].astype(np.uint8), kernel1, iterations=1)
            dilate_img2 = cv2.dilate(font_hint_fg[..., 0].astype(np.uint8), kernel2, iterations=1)
            dilate_text = dilate_img2 - dilate_img1
            result = (font_hint_fg[..., 0]-font_hint_bg + dilate_text).clip(0, 255)
            font_hint_bg[font_hint_fg[..., 0] > 0] = 0
            result = (result + font_hint_bg).clip(0, 255)
            font_hint_bg = result[..., None]
            item_dict['font_hint'] = font_hint_bg/255.
        else:
            item_dict['font_hint'] = np.zeros((512, 512, 3))
    return item_dict


def process(model, ddim_sampler, item_dict, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, strength, scale, seed, eta):
    with torch.no_grad():
        # text_prompt = item_dict['text_caption'] # 旧代码
        n_lines = item_dict['n_lines']
        pos_imgs = item_dict['positions']
        glyphs = item_dict['glyphs']
        gly_line = item_dict['gly_line']
        colors = item_dict['color']
        hint = np.sum(pos_imgs, axis=0).clip(0, 1)
        H, W, = (512, 512)
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)
        if save_memory:
            model.low_vram_shift(is_diffusing=False)
        info = {}
        info['glyphs'] = []
        info['gly_line'] = []
        info['positions'] = []
        info['n_lines'] = [n_lines]*num_samples
        info['colors'] = colors
        for i in range(n_lines):
            glyph = glyphs[i]
            pos = pos_imgs[i]
            gline = gly_line[i]
            info['glyphs'] += [arr2tensor(glyph, num_samples)]
            info['gly_line'] += [arr2tensor(gline, num_samples)]
            info['positions'] += [arr2tensor(pos, num_samples)]
            info['colors'][i] = arr2tensor(info['colors'][i], num_samples)/255.
        # get masked_x
        ref_img = np.zeros((H, W, 3))
        masked_img = ((ref_img.astype(np.float32) / 127.5) - 1.0 - hint*10).clip(-1, 1)

        masked_img = np.transpose(masked_img, (2, 0, 1))
        masked_img = torch.from_numpy(masked_img.copy()).float().cuda()
        if use_fp16:
            masked_img = masked_img.half()
        encoder_posterior = model.encode_first_stage(masked_img[None, ...])
        masked_x = model.get_first_stage_encoding(encoder_posterior).detach()
        if use_fp16:
            masked_x = masked_x.half()
        info['masked_x'] = torch.cat([masked_x for _ in range(num_samples)], dim=0)

        hint = arr2tensor(hint, num_samples)
        info['font_hint'] = arr2tensor(item_dict['font_hint'], num_samples)

        # --- MODIFIED: 构建灵活的上下文提示词 ---
        
        # 1. --- 条件提示词 (Conditional) ---
        text_prompt_cond = item_dict['text_caption']
        img_prompts_flat_list_cond = []
        for key in CONTEXT_PROMPT_KEYS:
            # 从 item_dict 获取提示词, 如果缺少则回退到旧的 'img_caption'
            prompt = item_dict.get(key, item_dict['img_caption']) 
            full_img_prompt = prompt + ', ' + a_prompt
            img_prompts_flat_list_cond.extend([full_img_prompt] * num_samples)
            
        text_prompts_list_cond = [text_prompt_cond] * num_samples
        # 构建 c_crossattn 列表: [ flat_img_prompts, text_prompts ]
        c_crossattn_cond = [img_prompts_flat_list_cond, text_prompts_list_cond]
        # 调用 get_learned_conditioning
        cond = model.get_learned_conditioning(dict(c_concat=[hint], c_crossattn=[c_crossattn_cond], text_info=info))

        # 2. --- 无条件提示词 (Un-conditional) ---
        text_prompt_uncond = ''
        img_prompts_flat_list_uncond = []
        for _ in CONTEXT_PROMPT_KEYS:
            img_prompts_flat_list_uncond.extend([n_prompt] * num_samples)
            
        text_prompts_list_uncond = [text_prompt_uncond] * num_samples
        c_crossattn_uncond = [img_prompts_flat_list_uncond, text_prompts_list_uncond]
        
        un_cond = model.get_learned_conditioning(dict(c_concat=[hint], c_crossattn=[c_crossattn_uncond], text_info=info))
        # --- End MODIFIED ---

        shape = (4, H // 8, W // 8)
        if save_memory:
            model.low_vram_shift(is_diffusing=True)
        model.control_scales = ([strength] * 13)
        tic = time.time()
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)
        cost = (time.time() - tic)*1000.
        if save_memory:
            model.low_vram_shift(is_diffusing=False)
        if use_fp16:
            samples = samples.half()
        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
        results += [cost]
    return results


if __name__ == '__main__':
    args = parse_args()
    times = []
    data_list = load_data(args.input_json)
    if not data_list: # 如果 data_list 为空，则退出
        print("No data loaded. Exiting.")
        sys.exit(1)
        
    model = create_model(args.config_yaml, use_fp16=use_fp16).cuda().eval()
    if use_fp16:
        model = model.half()
    
    # --- NEW: 告知模型当前处于哪个阶段 (推理总是=阶段2) ---
    # 这对于 cldm.py 中的 __init__ 逻辑很重要
    model.training_stage = 2 
    # --- End NEW ---

    model.load_state_dict(load_state_dict(args.ckpt_path, location='cuda'), strict=False)
    ddim_sampler = DDIMSampler(model)
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    for i in tqdm(range(len(data_list)), desc='generator'):
        item_dict = get_item(data_list, i)
        img_name = item_dict['img_name'].split('.')[0] + '_3.jpg'
        if os.path.exists(os.path.join(args.output_dir, img_name)):
            continue
        try:
            results = process(model, ddim_sampler, item_dict, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, strength, scale, seed, eta)
            times += [results.pop()]
            for idx, img in enumerate(results):
                # --- FIX: 修复拼写错误 item_dim -> item_dict ---
                img_name = item_dict['img_name'].split('.')[0]+f'_{idx}' + '.jpg'
                cv2.imwrite(os.path.join(args.output_dir, img_name), img[..., ::-1])
        except Exception as e:
            print(f"Error processing item {i} ({item_dict['img_name']}): {e}")
            # 可以在这里选择是 'continue' 还是 'break'
            continue

    if len(times) > 1:
        times = times[1:] # 忽略第一次加载的耗时
        print(f'Mean Time: {np.mean(times)/1000.:.2f} s.')
    elif len(times) == 1:
        print(f'Total Time for 1 item: {times[0]/1000.:.2f} s.')
    else:
        print("No samples were processed successfully.")
    print(f"Results saved to: {args.output_dir}")

