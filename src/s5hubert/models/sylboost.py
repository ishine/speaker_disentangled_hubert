# Copied and modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/hubert/modeling_hubert.py

# coding=utf-8
# Copyright 2021 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput

from .data2vec2 import Data2Vec2Config, Data2VecMultiModel
from .modules import init_module


class SylBoostForSequenceClassification(nn.Module):
    """
    https://arxiv.org/abs/2410.04029

    mkdir models/sylboost
    wget -t 0 -c -P models/sylboost https://www.cs.utexas.edu/~harwath/model_checkpoints/syllable_lm/SylBoost_625Hz.pth
    """

    def __init__(
        self,
        model_name_or_path="models/sylboost/SylBoost_625Hz.pth",
        classifier_proj_size: int = 256,
        num_labels: int = 1251,
        segmentation_layer: int = -2,
    ):
        super().__init__()

        d2v2_model = Data2VecMultiModel(Data2Vec2Config())
        state_dict = torch.load(model_name_or_path, weights_only=False)
        d2v2_model.load_state_dict({k[len("model.") :]: v for k, v in state_dict["model_seg"].items()}, strict=False)

        self.hubert = d2v2_model
        self.projector = nn.Linear(self.hubert.config.embed_dim, classifier_proj_size)
        self.classifier = nn.Linear(classifier_proj_size, num_labels, bias=False)

        # Initialize weights and apply final processing
        self.reset_parameters()
        self.num_labels = num_labels
        self.segmentation_layer = segmentation_layer

        self.freeze_base_model()
        self.hubert.eval()

    def reset_parameters(self):
        init_module(self.projector)
        init_module(self.classifier)

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        self.hubert.requires_grad_(False)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> Tuple | SequenceClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        hidden_states, padding_mask = self.hubert(input_values, attention_mask, self.segmentation_layer)
        hidden_states = hidden_states[-1]
        hidden_states = self.projector(hidden_states)

        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)
