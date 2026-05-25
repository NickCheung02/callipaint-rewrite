'''
AnyText2: Visual Text Generation and Editing With Customizable Attributes
Paper: https://arxiv.org/abs/2411.15245
Code: https://github.com/tyxsspa/AnyText2
Copyright (c) Alibaba, Inc. and its affiliates.
'''
import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from t3_dataset import T3DataSet
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
import shutil

###修改1################################################
# --- 新增一个参数来控制训练阶段 ---
TRAINING_STAGE = 2  # 修改这里来切换阶段: 1 或 2
if TRAINING_STAGE == 1:
        print("--- LAUNCHING TRAINING STAGE 1: IMAGE GENERATION ---")
        resume_path = 'models/anytext_v2.0.ckpt'
else: 
        print("--- LAUNCHING TRAINING STAGE 2: TEXT-IN-IMAGE GENERATION ---")
        resume_path = '/home/610-zzy/AnyText2-main-Real0922-DoubleStage-FHS-4-Calli/checkpoints/lightning_logs/version_1/checkpoints/epoch=29-step=3270.ckpt'  # 使用阶段1训练得到的模型进行阶段2训练
###修改1结束############################################

USING_DLC = False
NUM_NODES = 1
# Configs
ckpt_path = None  # if not None, continue training task, will not load "resume_path"
# resume_path = './models/anytext_v2.0.ckpt'  # finetune from scratch, run tool_add_anytext.py to get this ckpt

config_path = './models_yaml/anytext2_sd15.yaml'
grad_accum = 2  # default 1
batch_size = 48  # default 6
logger_freq = 1000
learning_rate = 2e-5  # default 2e-5
mask_ratio = 0  # default 0.5, ratio of mask for inpainting(text editing task), set 0 to disable
wm_thresh = 1.0  # perentage of skip images with watermark from training
save_ckpt_top = 20

root_dir = './checkpoints'  # path for save checkpoints
dataset_percent = 1
save_steps = None  # step frequency of saving checkpoints
save_epochs = 10  # epoch frequency of saving checkpoints
max_epochs = 60  # default 60
# font
rand_font = True
font_hint_prob = 0.8  # set 0 will disable font hint
color_prob = 1.0
font_hint_area = [0.7, 1]  # reserved area on each line of font hint
font_hint_randaug = True

assert (save_steps is None) != (save_epochs is None)


if __name__ == '__main__':
    log_img = os.path.join(root_dir, 'image_log/train')
    if os.path.exists(log_img):
        try:
            shutil.rmtree(log_img)
        except OSError:
            pass
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(config_path).cpu()
    ###修改2################################################
    model.training_stage = TRAINING_STAGE # <--- 传入阶段参数

    if ckpt_path is None:
        model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
    model.learning_rate = learning_rate
    ###修改3################################################
    ###修改3结束############################################
    if TRAINING_STAGE == 1:
        model.sd_locked = False # 解锁UNet进行训练
    else:
        model.sd_locked = True  # 锁定UNet
    model.sd_locked = True
    model.only_mid_control = False
    model.unlockQKV = False

    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=save_steps,
        every_n_epochs=save_epochs,
        save_top_k=save_ckpt_top,
        monitor="global_step",
        mode="max",
    )


    json_paths=[
        # '/home/610-zzy/dataset_final_in_use/data1_9000data_nowords/data1_9000data.json',
        '/home/610-zzy/AnyText2-main-Real0922-DoubleStage-FHS-4/data/data2_newdata/data2_newdata.json',
        # '/home/610-zzy/dataset_final_in_use/data3_poem_data_0/data3_poem_data_0.json',
        '/home/610-zzy/dataset_final_in_use/data4_RESULTS2_WithFit/data4_RESULTS2_WithFit.json',
    ]

    if USING_DLC:
        json_paths = [i.replace('/data/vdb', '/mnt/data', 1) for i in json_paths]
    glyph_scale = model.control_model.glyph_scale
    # dataset = T3DataSet(json_paths, max_lines=5, max_chars=20, mask_pos_prob=1.0, mask_img_prob=mask_ratio, glyph_scale=glyph_scale,
    #                     percent=dataset_percent, debug=False, using_dlc=USING_DLC, wm_thresh=wm_thresh, render_glyph=True,
    #                     trunc_cap=128, rand_font=rand_font, font_hint_prob=font_hint_prob, font_hint_area=font_hint_area,
    #                     font_hint_randaug=font_hint_randaug, color_prob=color_prob)
    dataset = T3DataSet(json_paths, max_lines=5, max_chars=20, mask_pos_prob=1.0, mask_img_prob=mask_ratio, glyph_scale=glyph_scale,
                        percent=dataset_percent, debug=False, using_dlc=USING_DLC, wm_thresh=wm_thresh, render_glyph=True,
                        trunc_cap=128, rand_font=rand_font, font_hint_prob=font_hint_prob, font_hint_area=font_hint_area,
                        font_hint_randaug=font_hint_randaug, color_prob=color_prob,
                        training_stage=TRAINING_STAGE) # <-- 添加此参数
    dataloader = DataLoader(dataset, num_workers=8, persistent_workers=True, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    # trainer = pl.Trainer(gpus=-1, precision=32, max_epochs=max_epochs, num_nodes=NUM_NODES, accumulate_grad_batches=grad_accum, callbacks=[logger, checkpoint_callback], default_root_dir=root_dir, strategy='ddp')
    trainer = pl.Trainer(accelerator='cuda', precision=32, max_epochs=max_epochs, num_nodes=NUM_NODES, accumulate_grad_batches=grad_accum, callbacks=[logger, checkpoint_callback], default_root_dir=root_dir, enable_progress_bar=True,devices=1)
    # Train!
    trainer.fit(model, dataloader, ckpt_path=ckpt_path)
