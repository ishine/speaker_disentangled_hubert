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

from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from transformers import Wav2Vec2Processor
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.hubert.modeling_hubert import HubertModel

from ..utils.mincut import mincut_torch
from ..utils.misc import fix_random_seed
from .modules import init_module


class HubertForSyllableDiscovery(nn.Module):
    def __init__(
        self,
        checkpoint_path="facebook/hubert-base-ls960",
        quantizer1_path="models/hubert/quantizer1.npy",
        quantizer2_path="models/hubert/quantizer2.npy",
        segmentation_layer: int = 8,
        seed: int = 0,
    ):
        super().__init__()
        self.segmentation_layer = segmentation_layer

        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

        self.hubert = HubertModel.from_pretrained(checkpoint_path, weights_only=False)
        self.hubert.eval()

        self.register_buffer("quantizer1", torch.from_numpy(np.load(quantizer1_path)) if quantizer1_path else None)
        self.register_buffer("quantizer2", torch.from_numpy(np.load(quantizer2_path)) if quantizer2_path else None)

        fix_random_seed(seed)

    @torch.inference_mode()
    def get_hidden_states(self, input_values: torch.Tensor) -> torch.Tensor:
        input_values = input_values.cpu().numpy()
        input_values = self.processor(input_values, sampling_rate=16000, return_tensors="pt").input_values
        input_values = input_values.to(self.hubert.device)
        hidden_states = self.hubert(input_values, output_hidden_states=True).hidden_states
        return hidden_states[self.segmentation_layer].squeeze(0)

    def forward(self, input_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden_states = self.get_hidden_states(input_values)
        if self.quantizer1 is None or self.quantizer2 is None:
            return {"hidden_states": hidden_states}

        frame_similarity = hidden_states @ hidden_states.T
        boundary, segment_features, frame_boundary = mincut_torch(hidden_states)

        # deduplicated syllabic units
        units = torch.cdist(segment_features, self.quantizer1).argmin(1)
        units = self.quantizer2[units]

        # duplicated syllabic units
        durations = frame_boundary[:, 1] - frame_boundary[:, 0]
        duplicated_units = torch.repeat_interleave(units, durations)
        return {
            "units": units,
            "duplicated_units": duplicated_units,
            "boundary": boundary,
            "frame_boundary": frame_boundary,
            "hidden_states": hidden_states,
            "frame_similarity": frame_similarity,
            "durations": durations,
        }


class HubertForSequenceClassification(nn.Module):
    def __init__(
        self,
        model_name_or_path="facebook/hubert-base-ls960",
        classifier_proj_size: int = 256,
        num_labels: int = 1251,
        segmentation_layer: int = 8,
    ):
        super().__init__()
        self.hubert = HubertModel.from_pretrained(model_name_or_path, weights_only=False)
        self.projector = nn.Linear(self.hubert.config.hidden_size, classifier_proj_size)
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
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = True,
        return_dict: bool | None = True,
        labels: torch.Tensor | None = None,
    ) -> Tuple | SequenceClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[1][self.segmentation_layer]

        hidden_states = self.projector(hidden_states)
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            padding_mask = self.hubert._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
