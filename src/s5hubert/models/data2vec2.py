# MIT License
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
from functools import partial
from types import SimpleNamespace
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

sys.path.append("src/SyllableLM")
from ...SyllableLM.syllablelm.data2vec.models.modalities.audio import AudioEncoder
from ...SyllableLM.syllablelm.data2vec.models.modalities.base import D2vModalityConfig
from ...SyllableLM.syllablelm.data2vec.models.modalities.modules import AltBlock, BlockEncoder


class Data2Vec2Config(PretrainedConfig):
    def __init__(
        self,
        _name="data2vec_multi",
        loss_beta=0.0,
        loss_scale=None,
        depth=8,
        start_drop_path_rate=0.0,
        end_drop_path_rate=0.0,
        num_heads=12,
        norm_eps=1e-05,
        norm_affine=True,
        encoder_dropout=0.1,
        post_mlp_drop=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        dropout_input=0.0,
        layerdrop=0.05,
        embed_dim=768,
        mlp_ratio=4.0,
        layer_norm_first=False,
        average_top_k_layers=8,
        end_of_block_targets=False,
        clone_batch=8,
        layer_norm_target_layer=False,
        batch_norm_target_layer=False,
        instance_norm_target_layer=True,
        instance_norm_targets=False,
        layer_norm_targets=False,
        ema_decay=0.999,
        ema_same_dtype=True,
        log_norms=True,
        ema_end_decay=0.99999,
        ema_anneal_end_step=75000,
        ema_encoder_only=False,
        max_update=400000,
        modalities={
            "audio": {
                "prenet_depth": 4,
                "prenet_layerdrop": 0.05,
                "prenet_dropout": 0.1,
                "start_drop_path_rate": 0.0,
                "end_drop_path_rate": 0.0,
                "num_extra_tokens": 0,
                "init_extra_token_zero": True,
                "mask_noise_std": 0.01,
                "mask_prob_min": None,
                "mask_prob": 0.5,
                "inverse_mask": False,
                "mask_prob_adjust": 0.05,
                "keep_masked_pct": 0.0,
                "mask_length": 5,
                "add_masks": False,
                "remove_masks": False,
                "mask_dropout": 0.0,
                "encoder_zero_mask": True,
                "mask_channel_prob": 0.0,
                "mask_channel_length": 64,
                "ema_local_encoder": False,
                "local_grad_mult": 1.0,
                "use_alibi_encoder": True,
                "alibi_scale": 1.0,
                "learned_alibi": False,
                "alibi_max_pos": None,
                "learned_alibi_scale": False,
                "learned_alibi_scale_per_head": True,
                "learned_alibi_scale_per_layer": False,
                "num_alibi_heads": 12,
                "model_depth": 8,
                "decoder": None,
                "extractor_mode": "layer_norm",
                "feature_encoder_spec": "[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
                "conv_pos_width": 95,
                "conv_pos_groups": 16,
                "conv_pos_depth": 5,
                "conv_pos_pre_ln": False,
            }
        },
        shared_decoder=None,
        min_target_var=0.1,
        min_pred_var=0.01,
        mae_init=False,
        seed=1,
        skip_ema=False,
        cls_loss=0.0,
        recon_loss=0.0,
        d2v_loss=1.0,
        decoder_group=False,
        **kwargs,
    ):
        self._name = _name
        self.loss_beta = loss_beta
        self.loss_scale = loss_scale
        self.depth = depth
        self.start_drop_path_rate = start_drop_path_rate
        self.end_drop_path_rate = end_drop_path_rate
        self.num_heads = num_heads
        self.norm_eps = norm_eps
        self.norm_affine = norm_affine
        self.encoder_dropout = encoder_dropout
        self.post_mlp_drop = post_mlp_drop
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.dropout_input = dropout_input
        self.layerdrop = layerdrop
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.layer_norm_first = layer_norm_first
        self.average_top_k_layers = average_top_k_layers
        self.end_of_block_targets = end_of_block_targets
        self.clone_batch = clone_batch
        self.layer_norm_target_layer = layer_norm_target_layer
        self.batch_norm_target_layer = batch_norm_target_layer
        self.instance_norm_target_layer = instance_norm_target_layer
        self.instance_norm_targets = instance_norm_targets
        self.layer_norm_targets = layer_norm_targets
        self.ema_decay = ema_decay
        self.ema_same_dtype = ema_same_dtype
        self.log_norms = log_norms
        self.ema_end_decay = ema_end_decay
        self.ema_anneal_end_step = ema_anneal_end_step
        self.ema_encoder_only = ema_encoder_only
        self.max_update = max_update
        self.modalities = modalities
        self.shared_decoder = shared_decoder
        self.min_target_var = min_target_var
        self.min_pred_var = min_pred_var
        self.mae_init = mae_init
        self.seed = seed
        self.skip_ema = skip_ema
        self.cls_loss = cls_loss
        self.recon_loss = recon_loss
        self.d2v_loss = d2v_loss
        self.decoder_group = decoder_group
        super().__init__(**kwargs)


class Data2VecMultiModel(PreTrainedModel):
    config_class = Data2Vec2Config

    def make_modality_encoder(
        self,
        cfg: D2vModalityConfig,
        embed_dim: int,
        make_block: Callable[[float], nn.ModuleList],
        norm_layer: Callable[[int], nn.LayerNorm],
        layer_norm_first: bool,
        alibi_biases,
        task,
    ) -> AudioEncoder:
        return AudioEncoder(
            cfg,
            embed_dim,
            make_block,
            norm_layer,
            layer_norm_first,
            alibi_biases,
            task,
        )

    def __init__(self, config: Data2Vec2Config, task=None):
        super().__init__(config)
        self.config = config

        make_layer_norm = partial(nn.LayerNorm, eps=config.norm_eps, elementwise_affine=config.norm_affine)

        def make_block(drop_path, dim=None, heads=None):
            return AltBlock(
                config.embed_dim if dim is None else dim,
                config.num_heads if heads is None else heads,
                config.mlp_ratio,
                qkv_bias=True,
                drop=config.encoder_dropout,
                attn_drop=config.attention_dropout,
                mlp_drop=config.activation_dropout,
                post_mlp_drop=config.post_mlp_drop,
                drop_path=drop_path,
                norm_layer=make_layer_norm,
                layer_norm_first=config.layer_norm_first,
                ffn_targets=not config.end_of_block_targets,
            )

        self.alibi_biases = {}
        self.modality_encoders = nn.ModuleDict()
        mod_cfg = SimpleNamespace(**config.modalities["audio"])
        self.modality_encoders["AUDIO"] = self.make_modality_encoder(
            mod_cfg,
            config.embed_dim,
            make_block,
            make_layer_norm,
            config.layer_norm_first,
            self.alibi_biases,
            task,
        )

        dpr = np.linspace(config.start_drop_path_rate, config.end_drop_path_rate, config.depth)

        self.blocks = nn.ModuleList([make_block(dpr[i]) for i in range(config.depth)])

        for pn, p in self.named_parameters():
            if len(p.shape) == 1 or pn.endswith(".bias") or "alibi_scale" in pn:
                p.optim_overrides = {"optimizer": {"weight_decay_scale": 0}}

    def forward(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        output_layer: Optional[int] = None,
    ):
        feature_extractor = self.modality_encoders["AUDIO"]

        x = feature_extractor.local_features(source)

        if padding_mask is not None:
            padding_mask = feature_extractor.convert_padding_mask(x, padding_mask)

        orig_B, orig_T, _ = x.shape

        x = x + feature_extractor.relative_positional_encoder(x)

        alibi_bias = None
        alibi_scale = feature_extractor.alibi_scale

        if feature_extractor.get_alibi_bias is not None:
            alibi_bias = feature_extractor.get_alibi_bias(
                batch_size=orig_B,
                time_steps=orig_T,
                heads=feature_extractor.modality_cfg.num_alibi_heads,
                dtype=torch.float32,
                device=x.device,
            )

            if alibi_scale is not None:
                alibi_scale = alibi_scale.clamp_min(0)
                if alibi_scale.size(0) == 1:
                    alibi_bias = alibi_bias * alibi_scale.squeeze(0).type_as(alibi_bias)
                    alibi_scale = None

        x = feature_extractor.context_encoder(
            x,
            padding_mask,
            alibi_bias,
            alibi_scale[: feature_extractor.modality_cfg.prenet_depth] if alibi_scale is not None else None,
        )

        alibi_scale = (
            alibi_scale[feature_extractor.modality_cfg.prenet_depth :]
            if alibi_scale is not None and alibi_scale.size(0) > 1
            else alibi_scale
        )

        layer_results = []
        for i, blk in enumerate(self.blocks):
            if not self.training or self.config.layerdrop == 0 or (np.random.random() > self.config.layerdrop):
                ab = alibi_bias
                if ab is not None and alibi_scale is not None:
                    scale = alibi_scale[i] if alibi_scale.size(0) > 1 else alibi_scale.squeeze(0)
                    ab = ab * scale.type_as(ab)

                x, lr = blk(x, padding_mask=padding_mask, alibi_bias=ab)
                layer_results.append(x)
                if output_layer is not None and i == len(self.blocks) + output_layer:
                    break

        return layer_results, padding_mask

    def freeze_pretrained_modules(self):
        # CNN
        self.modality_encoders["AUDIO"].requires_grad_(False)

        # Transformer
        self.modality_encoders["AUDIO"].context_encoder.requires_grad_(False)
        self.blocks.requires_grad_(False)

        self.modality_encoders["AUDIO"].alibi_scale.requires_grad_(False)
