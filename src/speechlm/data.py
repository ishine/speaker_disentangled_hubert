import random
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torchaudio
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from ..s5hubert import S5HubertForSyllableDiscovery


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, wav_paths):
        self.wav_paths = list(wav_paths)

    def __len__(self) -> int:
        return len(self.wav_paths)

    def __getitem__(self, n: int) -> Dict[str, Any]:
        wav_path = self.wav_paths[n]
        name = wav_path.stem
        wav_path = str(wav_path)
        input_values, sr = torchaudio.load(wav_path)
        input_values = input_values.squeeze(0)
        return {"input_values": input_values, "name": name}

    @staticmethod
    def collate_fn(batch):
        input_values = [item["input_values"] for item in batch]
        attention_mask = [torch.ones_like(item["input_values"], dtype=torch.long) for item in batch]
        names = [item["name"] for item in batch]

        input_values = pad_sequence(input_values, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        wavs_len = torch.tensor([len(item["input_values"]) for item in batch])

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "wavs_len": wavs_len,
            "padding_mask": ~attention_mask.bool(),
            "names": names,
        }


def get_collate_fn(
    tokenizer,
    units_per_sample: Optional[int] = None,
):
    def collate_fn(batch) -> Dict[str, torch.LongTensor]:
        input_ids = []
        names = []

        for item in batch:
            units = item["units"]

            if units_per_sample:
                diff = len(units) - units_per_sample

                if diff > 0:
                    start = random.randrange(diff)
                    units = units[start : start + units_per_sample]

            input_ids.append("".join([f"<{unit}>" for unit in units]))
            names.append(item["id"])

        inputs = tokenizer(input_ids)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        labels = input_ids.masked_fill(attention_mask.bool().logical_not(), -100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "names": names,
        }

    return collate_fn


def _tokenize(
    encoder,
    data_loader: torch.utils.data.DataLoader,
):
    def generate_dataset():
        for item in tqdm(data_loader):
            outputs = encoder(item["input_values"].cuda())

            yield {"id": item["name"][0], "units": outputs[0]["units"].tolist()}

    features = Features(
        {
            "id": Value("string"),
            "units": Sequence(Value("int32")),
        }
    )

    return Dataset.from_generator(generate_dataset, features=features)


def tokenize_slm21(config):
    app_dir = Path(config.dataset.APP_DIR).expanduser()

    swuggy_dir = app_dir / "datasets/sLM21-dataset/lexical"
    sblimp_dir = app_dir / "datasets/sLM21-dataset/syntactic"

    swuggy_dev_paths = list(swuggy_dir.glob("dev/*.wav"))
    sblimp_dev_paths = list(sblimp_dir.glob("dev/*.wav"))
    swuggy_test_paths = list(swuggy_dir.glob("test/*.wav"))
    sblimp_test_paths = list(sblimp_dir.glob("test/*.wav"))

    swuggy_dev_set = SpeechDataset(swuggy_dev_paths)
    sblimp_dev_set = SpeechDataset(sblimp_dev_paths)
    swuggy_test_set = SpeechDataset(swuggy_test_paths)
    sblimp_test_set = SpeechDataset(sblimp_test_paths)

    swuggy_dev_loader = torch.utils.data.DataLoader(swuggy_dev_set)
    sblimp_dev_loader = torch.utils.data.DataLoader(sblimp_dev_set)
    swuggy_test_loader = torch.utils.data.DataLoader(swuggy_test_set)
    sblimp_test_loader = torch.utils.data.DataLoader(sblimp_test_set)

    encoder = S5HubertForSyllableDiscovery.from_pretrained(config.speech2unit.model_name_or_path).cuda()

    swuggy_dev = _tokenize(encoder, swuggy_dev_loader)
    sblimp_dev = _tokenize(encoder, sblimp_dev_loader)
    swuggy_test = _tokenize(encoder, swuggy_test_loader)
    sblimp_test = _tokenize(encoder, sblimp_test_loader)

    swuggy = DatasetDict({"dev": swuggy_dev, "test": swuggy_test})
    sblimp = DatasetDict({"dev": sblimp_dev, "test": sblimp_test})

    swuggy.push_to_hub(config.dataset.swuggy)
    sblimp.push_to_hub(config.dataset.sblimp)


def tokenize_trainset(config, spk_ids: str = "1-9"):
    wav_dir_train = Path(config.dataset.wav_dir_train)
    train_paths = wav_dir_train.glob(f"*/[{spk_ids}]*/**/*" + config.dataset.ext_audio)
    train_set = SpeechDataset(train_paths)
    train_loader = torch.utils.data.DataLoader(train_set, num_workers=config.speech2unit.num_workers)

    encoder = S5HubertForSyllableDiscovery.from_pretrained(config.speech2unit.model_name_or_path).cuda()

    trainset = _tokenize(encoder, train_loader)
    trainset.push_to_hub(config.dataset.train, split=f"train{spk_ids}")
