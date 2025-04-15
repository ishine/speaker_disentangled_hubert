import sys
import warnings

import numpy as np
import torch
from scipy.spatial import distance
from torch.utils.data import ConcatDataset
from torchaudio.functional import edit_distance
from tqdm import tqdm

from ...sylber.sylber import Segmenter
from ..models.hubert import HubertForSyllableDiscovery
from ..models.s5hubert import S5HubertForSyllableDiscovery
from ..models.vghubert import VGHubertForSyllableDiscovery
from ..utils.data import LibriSpeech

sys.path.append("src/sdhubert")
from ...sdhubert.model.segmenter import MincutWrapper, SDHuBERTSegmenter

sys.path.append("src/SyllableLM")
from ...SyllableLM.extract_units import SylBoostFeatureReader

MODELS = {
    "hubert": HubertForSyllableDiscovery,
    "vghubert": VGHubertForSyllableDiscovery,
}


def eval_ued(config):
    if config.model.model_type.startswith("s5hubert"):
        model = S5HubertForSyllableDiscovery.load_pretrained(
            config.path.checkpoint,
            config.path.quantizer1,
            config.path.quantizer2,
            sec_per_syllable=config.mincut.sec_per_syllable,
            merge_threshold=config.mincut.merge_threshold,
            min_duration=config.mincut.min_duration,
            max_duration=config.mincut.max_duration,
        ).cuda()
    elif config.model.model_type == "sdhubert":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            segmenter = SDHuBERTSegmenter(
                config.path.checkpoint,
                layer=9,
                normcut_layer=11,
                normcut_threshold=2,
                device="cuda:0",
            )
        mincut = MincutWrapper(syl_dur=0.2, ft_sr=50)
        quantizer1 = np.load(config.path.quantizer1)
        quantizer2 = np.load(config.path.quantizer2)
    elif config.model.model_type in MODELS:
        model = MODELS[config.model.model_type](
            checkpoint_path=config.path.checkpoint,
            quantizer1_path=config.path.quantizer1,
            quantizer2_path=config.path.quantizer2,
            segmentation_layer=config.model.segmentation_layer,
        ).cuda()
    elif config.model.model_type == "sylboost":
        model = SylBoostFeatureReader(
            config.path.checkpoint,
            config.path.quantizer1,
            config.path.quantizer2,
            config.model.model_key,
        )
    elif config.model.model_type == "sylber":
        model = Segmenter(config.path.checkpoint)
        quantizer1 = np.load(config.path.quantizer1)
        quantizer2 = np.load(config.path.quantizer2)
    else:
        return

    dataset = ConcatDataset(
        [
            LibriSpeech(root=config.dataset.root, url="test-clean", max_sample_size=None),
            LibriSpeech(root=config.dataset.root, url="test-other", max_sample_size=None),
        ]
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=config.dataloader.num_workers,
        collate_fn=LibriSpeech.collate_fn,
    )

    total_ued = 0
    total_len = 0

    for batch in tqdm(data_loader):
        if config.model.model_type.startswith("s5hubert"):
            # original
            ref = model(
                input_values=batch["teacher_input_values"].cuda(),
                attention_mask=batch["teacher_attention_mask"].cuda(),
            )[0]["units"]

            # speaker perturbation
            hyp = model(
                input_values=batch["student_input_values"].cuda(),
                attention_mask=batch["student_attention_mask"].cuda(),
            )[0]["units"]
        elif config.model.model_type == "sdhubert":
            ref = mincut(**segmenter(batch["teacher_input_values"].squeeze(0).numpy()))
            hyp = mincut(**segmenter(batch["student_input_values"].squeeze(0).numpy()))
            ref = quantizer2[distance.cdist(ref["segment_features"], quantizer1).argmin(1)]
            hyp = quantizer2[distance.cdist(hyp["segment_features"], quantizer1).argmin(1)]
        elif config.model.model_type in MODELS:
            ref = model(batch["teacher_input_values"].cuda())["units"]
            hyp = model(batch["student_input_values"].cuda())["units"]
        elif config.model.model_type == "sylboost":
            ref = model.forward(batch["teacher_input_values"].cuda())["clusters_with_times"][0][0]
            hyp = model.forward(batch["student_input_values"].cuda())["clusters_with_times"][0][0]
        elif config.model.model_type == "sylber":
            ref = model(wav=batch["teacher_input_values"].squeeze(0).numpy())
            hyp = model(wav=batch["student_input_values"].squeeze(0).numpy())
            ref = quantizer2[distance.cdist(ref["segment_features"], quantizer1).argmin(1)]
            hyp = quantizer2[distance.cdist(hyp["segment_features"], quantizer1).argmin(1)]

        # unit edit distance (UED)
        # https://arxiv.org/abs/2209.15483
        total_ued += edit_distance(ref, hyp)
        total_len += len(ref)

    print(total_ued / total_len * 100)
