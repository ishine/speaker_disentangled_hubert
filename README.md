# S5-HuBERT: Self-Supervised Speaker-Separated Syllable HuBERT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ryota-komatsu/speaker_disentangled_hubert/blob/main/demo.ipynb)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2409.10103)
[![model](https://img.shields.io/badge/%F0%9F%A4%97-Models-blue)](https://huggingface.co/ryota-komatsu/s5-hubert)

This is the official repository of the IEEE SLT 2024 paper [Self-Supervised Syllable Discovery Based on Speaker-Disentangled HuBERT](https://arxiv.org/abs/2409.10103).

## Setup

```shell
conda create -y -n py310 -c pytorch -c nvidia -c conda-forge python=3.10.17 pip=24.0 faiss-gpu=1.10.0
conda activate py310
pip install -r requirements/requirements.txt

sh scripts/setup.sh
```

## Usage: encoding waveforms into pseudo-syllabic units

![](figures/usage.png)

```python
import torchaudio

from src.s5hubert import S5HubertForSyllableDiscovery

wav_path = "/path/to/wav"

# download a pretrained model from hugging face hub
model = S5HubertForSyllableDiscovery.from_pretrained("ryota-komatsu/s5-hubert").cuda()

# load a waveform
waveform, sr = torchaudio.load(wav_path)
waveform = torchaudio.functional.resample(waveform, sr, 16000)

# encode a waveform into pseudo-syllabic units
batch_outputs = model(waveform.cuda())

# pseudo-syllabic units
units = batch_outputs[0]["units"]  # [3950, 67, ..., 503]
```

## Demo

Google Colab demo is found [here](https://colab.research.google.com/github/ryota-komatsu/speaker_disentangled_hubert/blob/main/demo.ipynb).

## Models

![](figures/model.png)

You can download a pretrained model from [Hugging Face](https://huggingface.co/ryota-komatsu/s5-hubert).

Other models can be downloaded from [the old repository](https://huggingface.co/ryota-komatsu/speaker_disentangled_hubert/tree/main).

## Data Preparation

If you already have LibriSpeech, you can use it by editing [a config file](configs/default.yaml#L14);
```yaml
dataset:
  root: "/path/to/LibriSpeech/root" # ${dataset.root}/LibriSpeech/train-clean-100, train-clean-360, ...
```

otherwise you can download the new one under `dataset_root`.
```shell
dataset_root=data  # be consistent with dataset.root in a config file

sh scripts/download_librispeech.sh ${dataset_root}
```

Check the directory structure
```
dataset.root in a config file
└── LibriSpeech/
    ├── train-clean-100/
    ├── train-clean-360/
    ├── train-other-500/
    ├── dev-clean/
    ├── dev-other/
    ├── test-clean/
    ├── test-other/
    └── SPEAKERS.TXT
```

## Training & Evaluation

```shell
python main.py --config configs/speech2unit/default.yaml
```

To run only a sub-task (train, syllable_segmentation, quantize, or evaluate), specify it as an argument.

```shell
python main.py train --config configs/speech2unit/default.yaml
```

## Citation

```bibtex
@inproceedings{Komatsu_Self-Supervised_Syllable_Discovery_2024,
  author    = {Komatsu, Ryota and Shinozaki, Takahiro},
  title     = {Self-Supervised Syllable Discovery Based on Speaker-Disentangled HuBERT},
  year      = {2024},
  month     = {Dec.},
  booktitle = {IEEE Spoken Language Technology Workshop},
  pages     = {1131--1136},
  doi       = {10.1109/SLT61566.2024.10832325},
}
```