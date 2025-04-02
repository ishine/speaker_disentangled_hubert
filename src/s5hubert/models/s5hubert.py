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

from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.hubert.modeling_hubert import HubertEncoderLayer, HubertModel, HubertPreTrainedModel

from ..mincut.mincut_utils import mincut_torch
from ..utils.misc import fix_random_seed
from .modules import MLP, init_module


class S5Hubert(nn.Module):
    def __init__(
        self,
        model_name_or_path="facebook/hubert-base-ls960",
        init_last_layer: int = 3,
        head_out_size: int = 256,
        head_hidden_size: int = 2048,
        ema_decay: float = 0.999,
    ):
        super().__init__()
        self.ema_decay = ema_decay
        self.init_last_layer = init_last_layer

        self.student = HubertModel.from_pretrained(model_name_or_path, weights_only=False)
        self.student_projector = MLP(
            self.student.config.hidden_size,
            head_out_size,
            head_hidden_size,
        )
        self.student_predictor = MLP(
            head_out_size,
            head_out_size,
            head_hidden_size,
            norm_outputs=True,
        )
        self.loss_fn = nn.MSELoss()

        self.reset_parameters(init_last_layer)
        self.make_teacher(head_out_size, head_hidden_size)

    def reset_parameters(self, init_last_layer: int = 3):
        for m in self.student.encoder.layers[-init_last_layer:].modules():
            init_module(m)

        for m in self.student_projector.modules():
            init_module(m)

        for m in self.student_predictor.modules():
            init_module(m)

    def make_teacher(
        self,
        head_out_size: int = 256,
        head_hidden_size: int = 2048,
    ):
        self.teacher_encoder_layers = nn.ModuleList(
            [HubertEncoderLayer(self.student.config) for _ in range(self.student.config.num_hidden_layers)]
        )
        self.teacher_encoder_layers.load_state_dict(self.student.encoder.layers.state_dict())
        self.teacher_encoder_layers.requires_grad_(False)

        self.teacher_projector = MLP(
            self.student.config.hidden_size,
            head_out_size,
            head_hidden_size,
            norm_outputs=True,
        )
        self.teacher_projector.load_state_dict(self.student_projector.state_dict())
        self.teacher_projector.requires_grad_(False)

    @torch.no_grad()
    def update_teacher(self):
        for param_s, param_t in zip(self.student.encoder.layers.parameters(), self.teacher_encoder_layers.parameters()):
            param_t.data.mul_(self.ema_decay).add_((1 - self.ema_decay) * param_s.detach().data)

        for param_s, param_t in zip(self.student_projector.parameters(), self.teacher_projector.parameters()):
            param_t.data.mul_(self.ema_decay).add_((1 - self.ema_decay) * param_s.detach().data)

    def forward(
        self,
        teacher_input_values: torch.Tensor,
        student_input_values: torch.Tensor,
        teacher_attention_mask: Optional[torch.Tensor] = None,
        student_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # disable dropout
        self.student.feature_projection.eval()
        self.teacher_encoder_layers.eval()
        with torch.no_grad():
            teacher_hidden_states, teacher_padding_mask = self.teacher_forward(
                teacher_input_values, teacher_attention_mask
            )
            teacher_projection = self.teacher_projector(teacher_hidden_states[-1][teacher_padding_mask])

        # enable dropout
        self.student.feature_projection.train()
        self.student.encoder.layers.train()
        student_hidden_states, student_padding_mask = self.student_forward(student_input_values, student_attention_mask)
        student_projection = self.student_projector(student_hidden_states[-1][student_padding_mask])
        student_prediction = self.student_predictor(student_projection)

        return self.loss_fn(student_prediction, teacher_projection)

    def student_forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor], Optional[torch.Tensor]]:
        extract_features = self.student.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        feature_vector_attention_mask = None
        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self.student._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)
            feature_vector_attention_mask = attention_mask

        hidden_states = self.student.feature_projection(extract_features)

        all_hidden_states = ()

        if attention_mask is not None:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        position_embeddings = self.student.encoder.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.student.encoder.layer_norm(hidden_states)
        hidden_states = self.student.encoder.dropout(hidden_states)

        for layer in self.student.encoder.layers:
            all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = True if self.training and (dropout_probability < self.student.config.layerdrop) else False
            if not skip_the_layer:
                hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]

        all_hidden_states = all_hidden_states + (hidden_states,)

        return all_hidden_states, feature_vector_attention_mask

    def teacher_forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor], Optional[torch.Tensor]]:
        extract_features = self.student.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        feature_vector_attention_mask = None
        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self.student._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)
            feature_vector_attention_mask = attention_mask

        hidden_states = self.student.feature_projection(extract_features)

        all_hidden_states = ()

        if attention_mask is not None:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        position_embeddings = self.student.encoder.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.student.encoder.layer_norm(hidden_states)
        # hidden_states = self.student.encoder.dropout(hidden_states)

        for layer in self.teacher_encoder_layers:
            all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]

        all_hidden_states = all_hidden_states + (hidden_states,)

        return all_hidden_states, feature_vector_attention_mask

    def freeze_pretrained_modules(self):
        """for warmup"""
        # CNN
        self.student.feature_extractor._freeze_parameters()
        self.student.feature_projection.requires_grad_(False)

        # Transformer
        self.student.encoder.pos_conv_embed.requires_grad_(False)
        self.student.encoder.layer_norm.requires_grad_(False)
        self.student.encoder.layers.requires_grad_(False)
        self.student.encoder.layers[-self.init_last_layer :].requires_grad_(True)

    def defrost_transformer_encoder(self):
        # CNN
        self.student.feature_extractor._freeze_parameters()
        self.student.feature_projection.requires_grad_(False)

        # Transformer
        self.student.encoder.pos_conv_embed.requires_grad_(False)
        self.student.encoder.layer_norm.requires_grad_(False)
        self.student.encoder.layers.requires_grad_(True)


class S5HubertForSyllableDiscovery(HubertPreTrainedModel):
    def __init__(
        self,
        config,
        segmentation_layer: int = 8,
        n_units_step1: int = 16384,
        n_units_step2: int = 4096,
        seed: int = 0,
        max_chunk: int = 400080,  # 25 seconds
        deduplicate: bool = True,
    ):
        super().__init__(config)
        self.segmentation_layer = segmentation_layer
        self.n_units_step1 = n_units_step1
        self.n_units_step2 = n_units_step2
        self.sec_per_frame = np.prod(config.conv_stride) / 16000
        self.max_chunk = max_chunk
        self.deduplicate = deduplicate

        self.hubert = HubertModel(config)
        self.hubert.eval()

        self.register_buffer("quantizer1", torch.rand(n_units_step1, config.hidden_size))
        self.register_buffer("quantizer2", torch.zeros(n_units_step1, dtype=torch.int))

        fix_random_seed(seed)

    @classmethod
    def load_pretrained(cls, model_path, quantizer1_path, quantizer2_path) -> "S5HubertForSyllableDiscovery":
        """
        huggingface-cli login
        python

        from src.s5hubert import S5HubertForSyllableDiscovery

        model = S5HubertForSyllableDiscovery.load_pretrained(
            "models/s5-hubert",
            "models/s5-hubert/quantizer1.joblib",
            "models/s5-hubert/quantizer2.npy",
        )
        model.push_to_hub("s5-hubert", private=True)
        """
        model = cls.from_pretrained(model_path)
        model.quantizer1 = torch.from_numpy(joblib.load(quantizer1_path).cluster_centers_)
        model.quantizer2 = torch.from_numpy(np.load(quantizer2_path))
        return model

    def train_quantizer(
        self,
        hidden_states: np.ndarray,
        batch_size: int = 10000,
        verbose: int = 1,
        compute_labels: bool = False,
        random_state=0,
        max_no_improvement: int = 100,
        n_init: int = 5,
        reassignment_ratio: float = 0.0,
    ):
        quantizer1 = MiniBatchKMeans(
            n_clusters=self.n_units_step1,
            batch_size=batch_size,
            verbose=verbose,
            compute_labels=compute_labels,
            random_state=random_state,
            max_no_improvement=max_no_improvement,
            n_init=n_init,
            reassignment_ratio=reassignment_ratio,
        )
        quantizer1.fit(hidden_states)

        quantizer2 = AgglomerativeClustering(self.n_units_step2)
        quantizer2.fit_predict(quantizer1.cluster_centers_)

        self.register_buffer("quantizer1", torch.from_numpy(quantizer1.cluster_centers_))
        self.register_buffer("quantizer2", torch.from_numpy(quantizer2))

    def extract_features(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        extract_features = self.hubert.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        lengths = torch.full((extract_features.shape[0],), extract_features.shape[1], dtype=torch.int)
        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self.hubert._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)
            lengths = attention_mask.sum(dim=1)

        hidden_states = self.hubert.feature_projection(extract_features)

        if attention_mask is not None:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        position_embeddings = self.hubert.encoder.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.hubert.encoder.layer_norm(hidden_states)

        for layer in self.hubert.encoder.layers[: self.segmentation_layer]:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]

        return [h[:l] for h, l in zip(hidden_states, lengths)]

    @torch.inference_mode()
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sec_per_syllable: float = 0.2,
        merge_threshold: Optional[float] = 0.3,
        min_duration: int = 1,
        max_duration: int = 50,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Args:
            input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Raw speech waveform.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                1: non-mask
                0: mask
            sec_per_syllable (`float`):
                Seconds per syllable, used to predefine the number of syllables in the input speech.
            merge_threshold (`float`, *optional*):
                Merge threshold of the cosine similarity between adjacent syllabic segments.
            min_duration (`int`):
                The minimum unit duration, measured in frames.
            max_duration (`int`):
                The maximum unit duration, measured in frames, before adjacent segment merge.

        Returns:
            units (`torch.LongTensor`):
                Discrete pseudo-syllabic units.
            intermediate_units (`torch.LongTensor`):
                Intermediate K-means units.
            durations (`torch.LongTensor`):
                Durations of units, measured in frames.
            dense (`torch.FloatTensor` of shape `((sequence_length - 400) // 320 + 1, hidden_size)`):
                Latent speech frame representations extracted from the syllable segmentation layer.
        """
        outputs = []

        hidden_states = self.extract_features(input_values, attention_mask)

        for dense in hidden_states:
            _, segment_features, frame_boundary = mincut_torch(
                dense,
                sec_per_frame=self.sec_per_frame,
                sec_per_syllable=sec_per_syllable,
                merge_threshold=merge_threshold,
                min_duration=min_duration,
                max_duration=max_duration,
            )

            # K-means
            intermediate_units = torch.cdist(segment_features, self.quantizer1).argmin(1)

            # Agglomerative clustering on K-means centroids
            units = self.quantizer2[intermediate_units]

            durations = frame_boundary[:, 1] - frame_boundary[:, 0]

            if not self.deduplicate:
                units = torch.repeat_interleave(units, durations)
                intermediate_units = torch.repeat_interleave(intermediate_units, durations)

            outputs.append(
                {
                    "units": units,
                    "intermediate_units": intermediate_units,
                    "durations": durations,
                    "dense": dense,
                }
            )
        return outputs

    @torch.inference_mode()
    def chunk_forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sec_per_syllable: float = 0.2,
        merge_threshold: Optional[float] = 0.3,
        min_duration: int = 1,
        max_duration: int = 50,
        batch_size: int = 16,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        process a single long (e.g., 1 hour) speech.

        Args:
            input_values (`torch.FloatTensor` of shape `(1, sequence_length)`):
                Raw speech waveform.
            attention_mask (`torch.LongTensor` of shape `(1, sequence_length)`, *optional*):
                1: non-mask
                0: mask
            sec_per_syllable (`float`):
                Seconds per syllable, used to predefine the number of syllables in the input speech.
            merge_threshold (`float`, *optional*):
                Merge threshold of the cosine similarity between adjacent syllabic segments.
            min_duration (`int`):
                The minimum unit duration, measured in frames.
            max_duration (`int`):
                The maximum unit duration, measured in frames, before adjacent segment merge.
            batch_size (`int`):
                Batch size.

        Returns:
            units (`torch.LongTensor`):
                Discrete pseudo-syllabic units.
            intermediate_units (`torch.LongTensor`):
                Intermediate K-means units.
            durations (`torch.LongTensor`):
                Durations of units, measured in frames.
            dense (`torch.FloatTensor` of shape `((sequence_length - 400) // 320 + 1, hidden_size)`):
                Latent speech frame representations extracted from the syllable segmentation layer.
        """
        assert len(input_values) == 1

        outputs = []

        if attention_mask is None:
            attention_mask = torch.ones_like(input_values, dtype=torch.long)

        input_values = input_values.squeeze(0)
        attention_mask = attention_mask.squeeze(0)

        # split a long sequence into chunks
        input_values = torch.split(input_values, self.max_chunk)  # Tuple[torch.Tensor of shape `(len,)`]
        attention_mask = torch.split(attention_mask, self.max_chunk)  # Tuple[torch.Tensor of shape `(len,)`]

        input_values = pad_sequence(input_values, batch_first=True)  # (num_chunks, max_chunk)
        attention_mask = pad_sequence(attention_mask, batch_first=True)  # (num_chunks, max_chunk)

        # split chunks into batch
        input_values = torch.split(input_values, batch_size)
        attention_mask = torch.split(attention_mask, batch_size)

        hidden_states = []  # List[torch.Tensor]
        for batch_input_values, batch_attention_mask in zip(input_values, attention_mask):
            hidden_states += self.extract_features(batch_input_values, batch_attention_mask)

        for dense in hidden_states:
            _, segment_features, frame_boundary = mincut_torch(
                dense,
                sec_per_frame=self.sec_per_frame,
                sec_per_syllable=sec_per_syllable,
                merge_threshold=merge_threshold,
                min_duration=min_duration,
                max_duration=max_duration,
            )

            # K-means
            intermediate_units = torch.cdist(segment_features, self.quantizer1).argmin(1)

            # Agglomerative clustering on K-means centroids
            units = self.quantizer2[intermediate_units]

            durations = frame_boundary[:, 1] - frame_boundary[:, 0]

            if not self.deduplicate:
                units = torch.repeat_interleave(units, durations)
                intermediate_units = torch.repeat_interleave(intermediate_units, durations)

            outputs.append(
                {
                    "units": units,
                    "intermediate_units": intermediate_units,
                    "durations": durations,
                    "dense": dense,
                }
            )
        return outputs

    @torch.inference_mode()
    def get_hidden_states(self, input_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.hubert(input_values, output_hidden_states=True).hidden_states
        return hidden_states[self.segmentation_layer].squeeze(0)


class S5HubertForSequenceClassification(nn.Module):
    def __init__(
        self,
        model_name_or_path="models/s5-hubert",
        classifier_proj_size: int = 256,
        num_labels: int = 1251,
        segmentation_layer: int = 8,
    ):
        super().__init__()

        self.hubert = HubertModel.from_pretrained(model_name_or_path)
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
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
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
