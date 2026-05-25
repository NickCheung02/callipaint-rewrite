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
output_dir = '/data/vdb/yuxiang.tyx/AIGC/eval/gen_imgs_test'
# 修改 1: 将 num_samples 从 4 改为 1
num_samples = 1
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
font = ImageFont.truetype('font/Arial_Unicode.ttf', size=60)
glyph_scale = 1
use_fp16 = True
default_color = [500, 500, 500]
fonthint_type = 'Arial'

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
    for gt in content['data_list']:
        info = {}
        info['img_name'] = gt['img_name']
        info['caption'] = gt['caption']
        if PLACE_HOLDER in info['caption']:
            count += 1
            info['caption'] = info['caption'].replace(PLACE_HOLDER, " ")
        if 'annotations' in gt:
            polygons = []
            texts = []
            pos = []
            for annotation in gt['annotations']:
                if len(annotation['polygon']) == 0:
                    continue
                if annotation['valid'] is False:
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
    item_dict['img_caption'] = cur_item['caption']
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
        img_prompt = item_dict['img_caption']
        text_prompt = item_dict['text_caption']
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
        cond = model.get_learned_conditioning(dict(c_concat=[hint], c_crossattn=[[[img_prompt + ', ' + a_prompt] * num_samples, [text_prompt] * num_samples]], text_info=info))
        un_cond = model.get_learned_conditioning(dict(c_concat=[hint], c_crossattn=[[[n_prompt] * num_samples, [''] * num_samples]], text_info=info))
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
        
        # 修改 2: process 函数现在只返回一张图片和耗时
        # 原来的 results 是一个包含 num_samples 张图和 cost 的列表
        # 现在我们只取第一张图 (索引0)，并返回它和 cost
        single_image = x_samples[0] # 取出单张图片
        return single_image, cost # 返回 (image, cost)

if __name__ == '__main__':
    args = parse_args()
    times = []
    data_list = load_data(args.input_json)
    model = create_model(args.config_yaml, use_fp16=use_fp16).cuda().eval()
    if use_fp16:
        model = model.half()
    model.load_state_dict(load_state_dict(args.ckpt_path, location='cuda'), strict=False)
    ddim_sampler = DDIMSampler(model)
    
    for i in tqdm(range(len(data_list)), desc='generator'):
        item_dict = get_item(data_list, i)
        # 修改 3: 检查是否存在同名文件时，使用原始的 img_name
        img_name = item_dict['img_name'] # 直接使用原始文件名
        output_path = os.path.join(args.output_dir, img_name)
        
        # 跳过已存在的文件
        if os.path.exists(output_path):
            continue
            
        # 修改 4: 接收 process 函数返回的单张图片和耗时
        result_image, cost = process(model, ddim_sampler, item_dict, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, strength, scale, seed, eta)
        times.append(cost)
        
        # 修改 5: 将生成的单张图片保存为原始文件名
        cv2.imwrite(output_path, result_image[..., ::-1])
        
    times = times[1:] # 排除第一次（可能包含加载时间）
    print(f'Mean Time: {np.mean(times)/1000.:.2f} s.')