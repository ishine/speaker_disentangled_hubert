import sys
from pathlib import Path

import numpy as np
import torchaudio
from tqdm import tqdm

from ...sylber.sylber import Segmenter
from ..mincut.mincut_utils import parallel_mincut
from ..models.hubert import HubertForSyllableDiscovery
from ..models.s5hubert import S5HubertForSyllableDiscovery
from ..models.vghubert import VGHubertForSyllableDiscovery

sys.path.append("src/SyllableLM")
from ...SyllableLM.extract_units import SylBoostFeatureReader

MODELS = {
    "hubert": HubertForSyllableDiscovery,
    "vghubert": VGHubertForSyllableDiscovery,
}


def syllable_segmentation(config):
    if config.model.model_type.startswith("s5hubert"):
        model = S5HubertForSyllableDiscovery.from_pretrained(
            config.path.checkpoint,
            segmentation_layer=config.model.segmentation_layer,
            sec_per_syllable=config.mincut.sec_per_syllable,
            merge_threshold=config.mincut.merge_threshold,
            min_duration=config.mincut.min_duration,
            max_duration=config.mincut.max_duration,
        ).cuda()
    elif config.model.model_type in MODELS:
        model = MODELS[config.model.model_type](
            checkpoint_path=config.path.checkpoint,
            quantizer1_path=None,
            quantizer2_path=None,
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
    else:
        return

    wav_dir = Path(config.dataset.root) / "LibriSpeech"
    segment_dir = Path(config.path.segment_dir)
    segment_paths = []
    files = [
        config.dataset.train_file,
        config.dataset.dev_file,
        config.dataset.test_file,
    ]

    for file in files:
        with open(file) as f:
            lines = f.readlines()
            for wav_name in tqdm(lines, disable=config.common.disable_tqdm):
                wav_name = wav_name.rstrip()
                wav_path = wav_dir / wav_name
                wav_path = str(wav_path)  # for sox backend
                wav, sr = torchaudio.load(wav_path)

                if config.model.model_type.startswith("s5hubert") or config.model.model_type in MODELS:
                    wav = wav.cuda()
                    hidden_states = model.get_hidden_states(wav).cpu().numpy()
                    outputs = {"hidden_states": hidden_states}
                elif config.model.model_type == "sylboost":
                    wav = wav.cuda()
                    outputs = model.forward(wav)
                    outputs = {
                        "segments": outputs["clusters_with_times"][0][1:].T * config.mincut.sec_per_frame,
                        "units": outputs["clusters_with_times"][0][0],
                    }
                elif config.model.model_type == "sylber":
                    wav = wav.squeeze(0).numpy()
                    outputs = model(wav=wav)

                # save hidden states
                segment_name = wav_name.replace(".flac", ".npy")
                segment_path = segment_dir / segment_name
                segment_path.parent.mkdir(parents=True, exist_ok=True)
                segment_paths.append(segment_path)
                np.save(segment_path, outputs)

    if config.model.model_type.startswith("s5hubert") or config.model.model_type in MODELS:
        parallel_mincut(
            segment_paths,
            config.common.disable_tqdm,
            config.mincut.sec_per_frame,
            config.mincut.sec_per_syllable,
            config.mincut.merge_threshold,
            config.mincut.min_duration,
            config.mincut.max_duration,
            config.mincut.num_workers,
        )
