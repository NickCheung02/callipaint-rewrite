'''
AnyText2: Visual Text Generation and Editing With Customizable Attributes
Paper: https://arxiv.org/abs/2411.15245
Code: https://github.com/tyxsspa/AnyText2
Copyright (c) Alibaba, Inc. and its affiliates.
'''
import sys
import os
import torch
from cldm.model import create_model

add_ocr = False  # merge OCR model
add_style_ocr = False  # if style_ocr_trainable=True should set to True
ocr_path = './ocr_weights/ppv3_rec.pth'


if len(sys.argv) == 3:
    input_path = sys.argv[1]
    output_path = sys.argv[2]
else:
    print('Args are wrong, using default input and output path!')
    input_path = '/home/610-zzy/AnyText2-main-Original2/models/Stable-diffusion/v1-5-pruned.ckpt'  # sd1.5, download it by your own
    output_path = './models/anytext2_sd15_scratch.ckpt'

assert os.path.exists(input_path), 'Input model does not exist.'
assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


model = create_model(config_path='./models_yaml/anytext2_sd15.yaml')

# pretrained_weights = torch.load(input_path)
pretrained_weights = torch.load(input_path, weights_only=False)

if 'state_dict' in pretrained_weights:
    pretrained_weights = pretrained_weights['state_dict']

scratch_dict = model.state_dict()

target_dict = {}
for k in scratch_dict.keys():
    is_control, name = get_node_name(k, 'control_')
    if is_control:
        copy_k = 'model.diffusion_' + name
    else:
        if 'attn1x' in k:
            copy_k = k.replace('attn1x', 'attn1')
        elif 'attn2x' in k:
            copy_k = k.replace('attn2x', 'attn2')
        else:
            copy_k = k
    if copy_k in pretrained_weights:
        target_dict[k] = pretrained_weights[copy_k].clone()
        # print(f'These weights are copied from pretrain model: {k}')
    else:
        target_dict[k] = scratch_dict[k].clone()
        print(f'These weights are newly added: {k}')

if add_ocr:
    ocr_weights = torch.load(ocr_path)
    if 'state_dict' in ocr_weights:
        ocr_weights = ocr_weights['state_dict']
    for key in ocr_weights:
        new_key = 'text_predictor.' + key
        target_dict[new_key] = ocr_weights[key]
    print('ocr weights are added!')

if add_style_ocr:
    ocr_weights = torch.load(ocr_path)
    if 'state_dict' in ocr_weights:
        ocr_weights = ocr_weights['state_dict']
    for key in ocr_weights:
        new_key = 'embedding_manager.font_predictor.' + key
        target_dict[new_key] = ocr_weights[key]
    print('ocr weights are added!')

model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), output_path)
print('Done.')
