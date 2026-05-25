import einops
import torch
import torch as th
import torch.nn as nn
import copy
from easydict import EasyDict as edict

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion, get_print_grad_hook, print_grad
from ldm.util import log_txt_as_img, exists, instantiate_from_config
# from ldm.models.diffusion.ddim import DDIMSampler
from cldm.ddim_hacked import DDIMSampler
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from .recognizer import TextRecognizer, create_predictor
from omegaconf.listconfig import ListConfig
import cv2


PRINT_DEBUG = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, 
                context_input=None, context_middle=None, context_output=None, # <-- 替换旧的 'context=None'
                control=None, only_mid_control=False, attnx_scale=1.0, **kwargs):
        
        # --- NEW: 定义 fallback 逻辑 ---
        # 如果某个上下文未提供，则使用另一个
        if context_input is None:
            context_input = context_middle if context_middle is not None else context_output
        if context_middle is None:
            context_middle = context_input # 回退到 input
        if context_output is None:
            context_output = context_input # 回退到 input
            
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        if self.use_fp16:
            t_emb = t_emb.half()
            
        if self.input_attnx:
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            # --- 1. 在 input_blocks 中使用 context_input ---
            for module in self.input_blocks:
                h = module(h, emb, context_input, attnx_scale)
                hs.append(h)
        else:
            with torch.no_grad():
                emb = self.time_embed(t_emb)
                h = x.type(self.dtype)
                # --- 1. 在 input_blocks 中使用 context_input ---
                for module in self.input_blocks:
                    h = module(h, emb, context_input, attnx_scale)
                    hs.append(h)

        if self.mid_attnx:
            # --- 2. 在 middle_block 中使用 context_middle ---
            h = self.middle_block(h, emb, context_middle, attnx_scale)
        else:
            with torch.no_grad():
                # --- 2. 在 middle_block 中使用 context_middle ---
                h = self.middle_block(h, emb, context_middle, attnx_scale)

        if control is not None:
            h += control.pop()

        # --- 3. 在 output_blocks 中使用 context_output ---
        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            # 将 context_output 传递给输出块
            h = module(h, emb, context_output, attnx_scale)
            
        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            glyph_channels,
            position_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
            fast_control=True,
            glyph_scale=1,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'
        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.use_fp16 = use_fp16
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        self.fast_control = fast_control
        self.glyph_channels = glyph_channels
        self.glyph_scale = glyph_scale

        if self.glyph_scale == 2:
            self.glyph_block = TimestepEmbedSequential(
                conv_nd(dims, glyph_channels, 8, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 8, 8, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 8, 16, 3, padding=1, stride=2),
                nn.SiLU(),
                conv_nd(dims, 16, 16, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 16, 32, 3, padding=1, stride=2),
                nn.SiLU(),
                conv_nd(dims, 32, 32, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 32, 96, 3, padding=1, stride=2),
                nn.SiLU(),
                conv_nd(dims, 96, 96, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 96, 256, 3, padding=1, stride=2),
                nn.SiLU(),
            )
        elif self.glyph_scale == 1:
            self.glyph_block = TimestepEmbedSequential(
                conv_nd(dims, glyph_channels, 16, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 16, 16, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 16, 32, 3, padding=1, stride=2),
                nn.SiLU(),
                conv_nd(dims, 32, 32, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 32, 96, 3, padding=1, stride=2),
                nn.SiLU(),
                conv_nd(dims, 96, 96, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 96, 256, 3, padding=1, stride=2),
                nn.SiLU(),
            )

        self.position_block = TimestepEmbedSequential(
            conv_nd(dims, position_channels, 8, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 8, 8, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 8, 16, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 64, 3, padding=1, stride=2),
            nn.SiLU(),
        )
        self.fuse_block_za = zero_module(conv_nd(dims, 256+64+4, model_channels, 3, padding=1))

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, text_info, timesteps, context, **kwargs):
        # guided_hint from text_info
        if self.fast_control:
            timesteps = torch.tensor([0]*hint.shape[0], device=hint.device).long()
        glyphs = torch.sum(torch.stack(text_info['glyphs']), dim=0)
        glyphs = (torch.sum(glyphs, dim=1) != 0).to(glyphs.dtype).unsqueeze(1)
        positions = torch.cat(text_info['positions'], dim=1).sum(dim=1, keepdim=True)
        enc_glyph = self.glyph_block(glyphs, None, None)
        enc_pos = self.position_block(positions, None, None)
        guided_hint = self.fuse_block_za(torch.cat([enc_glyph, enc_pos, text_info['masked_x']], dim=1))

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        if self.use_fp16:
            t_emb = t_emb.half()
        emb = self.time_embed(t_emb)
        outs = []

        if not self.fast_control:
            h = x.type(self.dtype)
        else:
            h = torch.zeros_like(x).to(x.device)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlLDM(LatentDiffusion):
    def __init__(self, control_stage_config, control_key, glyph_key, position_key, only_mid_control, 
                 loss_alpha=0, loss_beta=0, with_step_weight=False, use_vae_upsample=False, 
                 use_text_cond=False, use_text_emb=False, latin_weight=1.0, 
                 embedding_manager_config=None, training_stage=2,
                 context_injection_config=None, # <--- 新增的配置入口
                 *args, **kwargs):
        self.use_fp16 = kwargs.pop('use_fp16', False)
        super().__init__(*args, **kwargs)
        self.training_stage = training_stage
        self.control_key = control_key
        self.glyph_key = glyph_key
        self.position_key = position_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self.attnx_scale = 1.0
        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta
        self.with_step_weight = with_step_weight
        self.use_vae_upsample = use_vae_upsample
        self.use_text_cond = use_text_cond
        self.use_text_emb = use_text_emb
        self.latin_weight = latin_weight
        self.control = None
        self.control_uncond = None
        self.is_uncond = False

        # --- NEW: 定义提示词键和灵活的注入配置 ---
        # 1. 定义您要使用的图像提示词的键 (必须与 t3_dataset.py 中一致)
        self.context_prompt_keys = ['full_prompt', 'element_prompt', 'mood_prompt', 'style_prompt']
        
        # 2. 您的“灵活入口”：
        #    定义哪种提示词(键)注入到UNet的哪个部分(input, middle, output)
        #    您可以传入一个列表，模型会自动将它们拼接(torch.cat)
        #    默认回退配置
        default_config = {
            'input': ['style_prompt', 'mood_prompt'], # 示例：输入层使用 风格+意境
            'middle': ['mood_prompt'],                 # 示例：中间层使用 意境
            'output': ['element_prompt', 'full_prompt'] # 示例：输出层使用 元素+完整描述
        }
        self.context_injection_config = context_injection_config or default_config
        print("--- Context Injection Config (分层注入配置) ---")
        print(f"Input Blocks: {self.context_injection_config.get('input')}")
        print(f"Middle Block: {self.context_injection_config.get('middle')}")
        print(f"Output Blocks: {self.context_injection_config.get('output')}")
        print("---------------------------------------------")
        # --- NEW: 修改结束 ---

        if self.training_stage == 2:
            self.control_model = instantiate_from_config(control_stage_config)
            if embedding_manager_config is not None and embedding_manager_config.params.valid:
                self.embedding_manager = self.instantiate_embedding_manager(embedding_manager_config, self.cond_stage_model)
                for param in self.embedding_manager.embedding_parameters():
                    param.requires_grad = True
            else:
                self.embedding_manager = None
        else:
            self.control_model = None
            self.embedding_manager = None

        if self.loss_alpha > 0 or self.loss_beta > 0 or self.embedding_manager:
            if embedding_manager_config.params.emb_type == 'ocr':
                self.text_predictor = create_predictor().eval()
                args = edict()
                args.rec_image_shape = "3, 48, 320"
                args.rec_batch_num = 6
                args.rec_char_dict_path = './ocr_recog/ppocr_keys_v1.txt'
                args.use_fp16 = self.use_fp16
                self.cn_recognizer = TextRecognizer(args, self.text_predictor)
                for param in self.text_predictor.parameters():
                    param.requires_grad = False
                if self.embedding_manager:
                    self.embedding_manager.recog = self.cn_recognizer
            if 'add_style_ocr' in embedding_manager_config.params and embedding_manager_config.params.add_style_ocr and self.embedding_manager.style_encoder:
                self.embedding_manager.style_encoder.use_fp16 = self.use_fp16

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        # 1. 加载 x, control, mx (来自 LatentDiffusion in ddpm.py)
        # 注意：原版 get_input 返回 (x, c, mask), 我们需要全部
        x, c_original, mx = super().get_input(batch, self.first_stage_key, mask_k='masked_img', *args, **kwargs)
        
        # 2. 加载 control (来自原始 ControlLDM.get_input)
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()

        inv_mask = batch['inv_mask']
        if bs is not None:
            inv_mask = inv_mask[:bs]
        inv_mask = inv_mask.to(self.device)
        inv_mask = einops.rearrange(inv_mask, 'b h w c -> b c h w')
        inv_mask = inv_mask.to(memory_format=torch.contiguous_format).float()

        # 3. 加载所有 text_info (glyphs, positions, etc.)
        glyphs = copy.deepcopy(batch[self.glyph_key])
        gly_line = copy.deepcopy(batch['gly_line'])
        positions = copy.deepcopy(batch[self.position_key])
        colors = copy.deepcopy(batch['color'])
        n_lines = copy.deepcopy(batch['n_lines'])
        language = copy.deepcopy(batch['language'])
        texts = copy.deepcopy(batch['texts'])
        font_hint = copy.deepcopy(batch['font_hint'])

        if bs is not None:
            font_hint = font_hint[:bs]
        font_hint = font_hint.to(self.device)
        font_hint = einops.rearrange(font_hint, 'b h w c -> b c h w')
        font_hint = font_hint.to(memory_format=torch.contiguous_format).float()
        assert len(glyphs) == len(positions)
        for i in range(len(glyphs)):
            if bs is not None:
                glyphs[i] = glyphs[i][:bs]
                gly_line[i] = gly_line[i][:bs]
                positions[i] = positions[i][:bs]
                colors[i] = colors[i][:bs]
                n_lines = n_lines[:bs]
            glyphs[i] = glyphs[i].to(self.device)
            gly_line[i] = gly_line[i].to(self.device)
            positions[i] = positions[i].to(self.device)
            colors[i] = colors[i].to(self.device)
            glyphs[i] = einops.rearrange(glyphs[i], 'b h w c -> b c h w')
            gly_line[i] = einops.rearrange(gly_line[i], 'b h w c -> b c h w')
            positions[i] = einops.rearrange(positions[i], 'b h w c -> b c h w')
            glyphs[i] = glyphs[i].to(memory_format=torch.contiguous_format).float()
            gly_line[i] = gly_line[i].to(memory_format=torch.contiguous_format).float()
            positions[i] = positions[i].to(memory_format=torch.contiguous_format).float()
            colors[i] = colors[i].to(memory_format=torch.contiguous_format).float()/255.

        info = {}
        info['glyphs'] = glyphs
        info['positions'] = positions
        info['colors'] = colors
        info['n_lines'] = n_lines
        info['language'] = language
        info['texts'] = texts
        info['img'] = batch['img']  # nhwc, (-1,1)
        info['masked_x'] = mx
        info['gly_line'] = gly_line
        info['inv_mask'] = inv_mask
        info['font_hint'] = font_hint

        # --- 4. NEW: 打包所有图像提示词 ---
        if bs is None:
            bs = x.shape[0]
        
        img_prompts_flat_list = []
        # 打印调试信息：检查 keys 的内容
        # print(f"DEBUG: context_prompt_keys = {self.context_prompt_keys}")

        for key in self.context_prompt_keys:
            # 从batch获取提示词列表
            prompts = batch.get(key, [""] * bs)
            # [修复]：如果列表长度超过 bs，进行切片
            if isinstance(prompts, list) and len(prompts) >= bs:
                prompts = prompts[:bs]

            if len(prompts) != bs:
                prompts = [prompts[0]] * bs if len(prompts) == 1 else [""] * bs
            img_prompts_flat_list.extend(prompts)
        # 打印调试信息：检查生成的列表长度和 Batch Size
        # print(f"DEBUG: bs={bs}, flat_list_len={len(img_prompts_flat_list)}, expected={bs * len(self.context_prompt_keys)}")    
        # --- 5. NEW: 获取文本提示词 (用于 ControlNet) ---
        text_prompts_list = batch.get(self.cond_stage_key[1], [""] * bs) # Get text_caption
        # [修复]：同样对文本提示词进行切片
        if isinstance(text_prompts_list, list) and len(text_prompts_list) >= bs:
            text_prompts_list = text_prompts_list[:bs]
        
        # --- 6. NEW: 创建 c_crossattn ---
        # 我们将所有图像提示词打包到槽 [0]
        # 将文本提示词打包到槽 [1]
        c = [img_prompts_flat_list, text_prompts_list]
        
        return x, dict(c_crossattn=[c], c_concat=[control], text_info=info)

    def copy_tokens(self, all_embs, flag, init_vector):
        row_sums = flag.sum(dim=1)
        M = row_sums.max().item()
        M = max(M, 1)
        N = all_embs.shape[0]
        canvas = init_vector.expand(N, M, -1).clone()
        for i in range(N):
            if row_sums[i] > 0:
                true_indices = flag[i].nonzero(as_tuple=True)[0]
                canvas[i, :len(true_indices), :] = all_embs[i, true_indices, :].clone()
        return canvas

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        # --- NEW: 从 cond 字典中获取编码后的上下文 ---
        # (cond 是 get_learned_conditioning 的返回值)
        text_cond = cond.get('c_crossattn_text', None)
        img_contexts_dict = cond.get('c_crossattn_img_contexts', {})

        # --- NEW: 辅助函数，根据配置拼接上下文 ---
        def get_combined_context(block_type):
            # 1. 从您的“灵活入口”配置中获取此块要使用的键
            keys_to_use = self.context_injection_config.get(block_type, ['full_prompt'])

            # ======【验证代码】======
            # print(f"\n[DEBUG VERIFY] Block: {block_type}")
            # print(f"  -> Configured Keys: {keys_to_use}")
            # ======================

            # 2. 提取向量 (这部分逻辑保持不变，省略...)
            contexts_to_cat = []
            fallback_context = None
            if 'full_prompt' in img_contexts_dict:
                fallback_context = img_contexts_dict['full_prompt']
            elif len(img_contexts_dict) > 0:
                fallback_context = list(img_contexts_dict.values())[0]

            for key in keys_to_use:
                if key in img_contexts_dict:
                    contexts_to_cat.append(img_contexts_dict[key])
                else:
                    if fallback_context is not None:
                        contexts_to_cat.append(fallback_context)
            
            if not contexts_to_cat:
                if fallback_context is not None:
                    contexts_to_cat = [fallback_context]
                else:
                    return None

            # 3. 拼接 (修改这里的 return 为 result =)
            result = None 
            if len(contexts_to_cat) == 1:
                result = contexts_to_cat[0] # <--- 修改：赋值给 result
            else:
                unique_contexts = []
                seen_ids = set()
                for ctx in contexts_to_cat:
                    ctx_id = id(ctx)
                    if ctx_id not in seen_ids:
                        unique_contexts.append(ctx)
                        seen_ids.add(ctx_id)
                
                if len(unique_contexts) == 1:
                    result = unique_contexts[0] # <--- 修改：赋值给 result
                else:
                    result = torch.cat(unique_contexts, dim=1) # <--- 修改：赋值给 result

            # ======【现在这行代码可以被执行了】======
            # if result is not None:
                # print(f"  -> Final Context Shape: {result.shape} (seq_len={result.shape[1]})")
            
            return result

        # --- NEW: 生成三个UNet将使用的上下文向量 ---
        context_input = get_combined_context('input')
        context_middle = get_combined_context('middle')
        context_output = get_combined_context('output')

        # --- 阶段 1 (Stage 1) 逻辑更新 ---
        # 原本只使用 img_cond，现在我们传递所有分层上下文
        if self.training_stage == 1:
            return diffusion_model(x=x_noisy, timesteps=t, 
                                   context_input=context_input, 
                                   context_middle=context_middle, 
                                   context_output=context_output,
                                   attnx_scale=self.attnx_scale) # 确保也传递 attnx_scale
        
        # --- 阶段 2 (Stage 2) 逻辑 ---
        _hint = torch.cat(cond['c_concat'], 1)
        if self.use_fp16:
            x_noisy = x_noisy.half()
            
        if text_cond is None:
            control = None  # uncond
        else:
            if self.control is None or self.control_uncond is None or not self.control_model.fast_control:
                # ControlNet 使用 text_cond
                _control = self.control_model(x=x_noisy, timesteps=t, context=text_cond, hint=_hint, text_info=cond['text_info'])
                if not text_cond.requires_grad and self.control is not None and self.control_uncond is None:  # uncond
                    self.control_uncond = _control
                else:
                    self.control = _control
            if not text_cond.requires_grad:
                if self.is_uncond:
                    control = [c.clone() for c in self.control_uncond]
                    self.is_uncond = False
                else:
                    control = [c.clone() for c in self.control]
                    self.is_uncond = True
            else:
                control = [c.clone() for c in self.control]
            
            control = [c * scale for c, scale in zip(control, self.control_scales[:len(control)])]
            
        # --- NEW: 将所有分层上下文传递给 UNet ---
        eps = diffusion_model(x=x_noisy, timesteps=t, 
                              context_input=context_input, 
                              context_middle=context_middle, 
                              context_output=context_output, 
                              control=control, 
                              only_mid_control=self.only_mid_control, 
                              attnx_scale=self.attnx_scale)

        return eps

    def instantiate_embedding_manager(self, config, embedder):
        model = instantiate_from_config(config, embedder=embedder)
        return model

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        # --- NEW: 创建与 get_input 匹配的空提示词结构 ---
        # 1. 创建 (N * 4) 个空字符串用于图像提示词
        img_prompts_flat_list = [""] * (N * len(self.context_prompt_keys))
        # 2. 创建 N 个空字符串用于文本提示词
        text_prompts_list = [""] * N
        
        c_crossattn = [img_prompts_flat_list, text_prompts_list]
        # 3. text_info=None, c_concat=None (因为这是无条件)
        return self.get_learned_conditioning(dict(c_crossattn=[c_crossattn], text_info=None))

    def get_learned_conditioning(self, c):
        # c 是来自 get_input 的 dict: dict(c_crossattn=[c], c_concat=[control], text_info=info)
        # c['c_crossattn'][0] 是 [img_prompts_flat_list, text_prompts_list]

        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                
                text_info = c.get('text_info', None) # 确保 text_info 存在
                if self.embedding_manager is not None and text_info is not None:
                    self.embedding_manager.encode_text(text_info)
                
                cond_txt = c['c_crossattn'][0] # [img_flat_list, text_list]
                
                if self.embedding_manager is not None:
                    cond_txt_encoded = self.cond_stage_model.encode(cond_txt, embedding_manager=self.embedding_manager)
                else:
                    cond_txt_encoded = self.cond_stage_model.encode(cond_txt)
                
                # cond_txt_encoded 是 [encoded_img_flat, encoded_text]
                
                # --- NEW: 解包 (Un-flatten) 图像编码 ---
                encoded_img_flat = cond_txt_encoded[0] # Shape [bs * N_keys, 77, 768]
                encoded_text = cond_txt_encoded[1]     # Shape [bs, 77, 768]
                
                # 推断 batch size
                bs = encoded_text.shape[0]
                
                encoded_img_contexts = {}
                if encoded_img_flat is not None and encoded_img_flat.shape[0] > 0: # 确保非空
                    # 按照 N_keys * bs 的总长度，切分成 N_keys 个 [bs, 77, 768] 的块
                    chunks = torch.split(encoded_img_flat, bs, dim=0) 
                    
                    if len(chunks) == len(self.context_prompt_keys):
                        for i, key in enumerate(self.context_prompt_keys):
                            encoded_img_contexts[key] = chunks[i]
                    else:
                        print(f"Warning: Mismatch in prompt keys ({len(self.context_prompt_keys)}) and encoded chunks ({len(chunks)}). Falling back to full_prompt.")
                        # 回退：将所有编码都赋给 full_prompt (拼接)
                        encoded_img_contexts['full_prompt'] = encoded_img_flat
                
                # --- NEW: 将编码后的向量存入新的键 ---
                c['c_crossattn_img_contexts'] = encoded_img_contexts
                c['c_crossattn_text'] = encoded_text
                if 'c_crossattn' in c:
                    del c['c_crossattn'] # 删除旧键，避免混淆

                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
            
        self.control = None
        self.control_uncond = None
        self.is_uncond = False
        return c # c 现在包含 c_crossattn_img_contexts 和 c_crossattn_text

    def fill_caption(self, batch, place_holder='*'):
        bs = len(batch['n_lines'])
        cond_list = copy.deepcopy(batch[self.cond_stage_key[1]])
        for i in range(bs):
            n_lines = batch['n_lines'][i]
            if n_lines == 0:
                continue
            cur_cap = cond_list[i]
            for j in range(n_lines):
                r_txt = batch['texts'][j][i]
                cur_cap = cur_cap.replace(place_holder, f'"{r_txt}"', 1)
            cond_list[i] = cur_cap
        batch[self.cond_stage_key[1]] = cond_list

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        if self.cond_stage_trainable:
            with torch.no_grad():
                c = self.get_learned_conditioning(c)
        
        # --- NEW: 使用新的上下文键 ---
        # c 现在包含:
        # c['c_crossattn_img_contexts'] (字典)
        # c['c_crossattn_text'] (张量)
        # c['c_concat'] (列表)
        # c['text_info'] (字典)
        
        c_cat = c["c_concat"][0][:N]
        text_info = c["text_info"]
        text_info['glyphs'] = [i[:N] for i in text_info['glyphs']]
        text_info['gly_line'] = [i[:N] for i in text_info['gly_line']]
        text_info['positions'] = [i[:N] for i in text_info['positions']]
        text_info['n_lines'] = text_info['n_lines'][:N]
        text_info['masked_x'] = text_info['masked_x'][:N]
        text_info['img'] = text_info['img'][:N]

        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["masked_image"] = self.decode_first_stage(text_info['masked_x'])
        log["control"] = c_cat * 2.0 - 1.0
        log["img"] = text_info['img'].permute(0, 3, 1, 2)  # log source image if needed
        # get glyph
        glyph_bs = torch.stack(text_info['glyphs'])
        glyph_bs = torch.sum(glyph_bs, dim=0) * 2.0 - 1.0
        log["glyph"] = torch.nn.functional.interpolate(glyph_bs, size=(512, 512), mode='bilinear', align_corners=True,)
        # fill caption (使用 full_prompt)
        if not self.embedding_manager:
            self.fill_caption(batch)
        
        # --- NEW: 使用 full_prompt (或旧的 img_caption) 作为日志标题 ---
        captions = batch.get('full_prompt', batch.get('img_caption', [""] * N))
        log["conditioning"] = log_txt_as_img((512, 512), captions, size=16)


        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # --- NEW: 传递 'c' (它包含了所有新键) ---
            samples, z_denoise_row = self.sample_log(cond=c,
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            # --- NEW: 获取无条件 'uc' 字典 ---
            uc = self.get_unconditional_conditioning(N) # 'uc' 是包含新键的完整字典
            uc_cat = c_cat  # control (hint) 保持不变
            
            # --- NEW: 构建 uc_full (无条件) ---
            uc_full = {"c_concat": [uc_cat], 
                       "c_crossattn_img_contexts": uc['c_crossattn_img_contexts'],
                       "c_crossattn_text": uc['c_crossattn_text'],
                       "text_info": text_info} # text_info 保持不变 (用于ControlNet)
            
            # --- NEW: 构建 c_full (有条件) ---
            # (我们不能直接使用 'c'，因为它可能包含未切片的数据)
            c_img_contexts_sliced = {key: val[:N] for key, val in c['c_crossattn_img_contexts'].items()}
            c_text_sliced = c['c_crossattn_text'][:N]
            c_full = {"c_concat": [c_cat],
                      "c_crossattn_img_contexts": c_img_contexts_sliced,
                      "c_crossattn_text": c_text_sliced,
                      "text_info": text_info}
            
            # --- NEW: 使用 c_full 和 uc_full ---
            samples_cfg, tmps = self.sample_log(cond=c_full,
                                                batch_size=N, ddim=use_ddim,
                                                ddim_steps=ddim_steps, eta=ddim_eta,
                                                unconditional_guidance_scale=unconditional_guidance_scale,
                                                unconditional_conditioning=uc_full,
                                                )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
            pred_x0 = False  # wether log pred_x0
            if pred_x0:
                for idx in range(len(tmps['pred_x0'])):
                    pred_x0 = self.decode_first_stage(tmps['pred_x0'][idx])
                    log[f"pred_x0_{tmps['index'][idx]}"] = pred_x0

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, log_every_t=5, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
###修改4##############################################################
        if self.training_stage == 1:
            print("--- Training Stage 1: Optimizing UNet for image generation ---")
            # 阶段一：只优化 UNet
            params = list(self.model.diffusion_model.parameters())
            # 如果您也想微调文本编码器，可以取消下面这行注释
            # params += list(self.cond_stage_model.parameters()) 
        else: # training_stage == 2
            print("--- Training Stage 2: Optimizing ControlNet and EmbeddingManager for text-in-image generation ---")
            # 阶段二：只优化 ControlNet 和 EmbeddingManager (以及 AttnX 层)
            params = list(self.control_model.parameters())
            if self.embedding_manager:
                params += list(self.embedding_manager.embedding_parameters())
            
            # 添加 AttnX 层的参数
            for name, param in self.model.diffusion_model.named_parameters():
                if 'attn1x' in name or 'attn2x' in name:
                    params.append(param)
###修改4结束##########################################################
        # params = list(self.control_model.parameters())
        # if self.embedding_manager:
        #     params += list(self.embedding_manager.embedding_parameters())
        nCount = 0
        nParams = 0
        # for name, param in self.model.diffusion_model.named_parameters():
        #     if 'attn1x' in name or 'attn2x' in name:
        #         params += [param]
        #         nCount += 1
        #         nParams += param.numel()
        print(f'attnx layers are inserted, and {nCount} Wq, Wk,Wv or Wout.weight and Wout.bias are added to potimizers, param size = {nParams}!!!')
        if not self.sd_locked:
            # params += list(self.model.diffusion_model.input_blocks.parameters())
            # params += list(self.model.diffusion_model.middle_block.parameters())
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        if self.unlockQKV:
            nCount = 0
            nParams = 0
            for name, param in self.model.diffusion_model.named_parameters():
                if 'attn2.to_k' in name or 'attn2.to_v' in name or 'attn2.to_q' in name:
                    params += [param]
                    nCount += 1
                    # print(f'name={name}, params={param.numel()}')
                    nParams += param.numel()
            print(f'Cross attention is unlocked, and {nCount} Wq, Wk or Wv are added to potimizers, param size = {nParams}!!!')

        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
