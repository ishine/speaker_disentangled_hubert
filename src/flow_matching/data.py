import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torchaudio
from datasets import Array2D, Dataset, DatasetDict, Features, Sequence, Value
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from ..bigvgan.data import mel_spectrogram
from ..s5hubert import S5HubertForSyllableDiscovery


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        wav_dir: str,
        txt_dir: Optional[str] = None,
        split: str = "train-*",
        ext_audio: str = ".flac",
        ext_txt: Optional[str] = None,
    ):
        self.wav_dir = Path(wav_dir)
        self.txt_dir = Path(txt_dir) if txt_dir is not None else self.wav_dir
        self.wav_paths = sorted(self.wav_dir.glob(f"{split}/**/*" + ext_audio))

        self.ext_audio = ext_audio
        self.ext_txt = ext_txt

    def __len__(self) -> int:
        return len(self.wav_paths)

    def __getitem__(self, n: int) -> Dict[str, Any]:
        wav_path = self.wav_paths[n]
        wav_name = wav_path.relative_to(self.wav_dir)
        wav_name = wav_name.with_suffix("")
        wav_name = str(wav_name)
        wav_path = str(wav_path)

        input_values, sr = torchaudio.load(wav_path)
        input_values = torchaudio.functional.resample(input_values, sr, 16000)
        input_values = input_values.squeeze(0)

        return {"input_values": input_values, "name": wav_name}

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


class LibriTTS_R(SpeechDataset):
    def __init__(
        self,
        wav_dir,
        txt_dir=None,
        split: str = "train-*",
        ext_audio: str = ".wav",
        ext_txt: Optional[str] = ".normalized.txt",
    ):
        super().__init__(wav_dir, txt_dir, split, ext_audio, ext_txt)

    def __getitem__(self, n: int) -> Dict[str, Any]:
        item = super().__getitem__(n)

        txt_path = self.txt_dir / item["name"]
        txt_path = txt_path.with_suffix(self.ext_txt)

        transcript = ""
        if txt_path.is_file():
            with open(txt_path) as g:
                transcript = g.read().rstrip()

        item["transcript"] = transcript

        return item


class LibriSpeech(SpeechDataset):
    def __getitem__(self, n: int) -> Dict[str, Any]:
        item = super().__getitem__(n)

        # transcript
        split, speaker_id, chap_id, utterance_id = item["name"].split("/")
        file = self.txt_dir / split / speaker_id / chap_id / f"{speaker_id}-{chap_id}.trans.txt"

        with open(file) as f:
            for line in f:
                id, transcript = line.rstrip().split(" ", maxsplit=1)
                if id == utterance_id:
                    break

        item["transcript"] = transcript

        return item


def get_collate_fn(
    wav_dir: Optional[str] = None,
    ext_audio: str = ".wav",
):
    def parse_item(item: Dict[str, Any]):
        input_ids = item["units"] + 1  # 0: pad
        spectrogram_labels = item["spectrogram"]
        durations = item["durations"]
        transcript = item["transcript"]
        id = item["id"]
        wav = torch.zeros(1)  # dummy

        if wav_dir:
            wav_path = os.path.join(wav_dir, id + ext_audio)
            wav, sr = torchaudio.load(wav_path)
            wav = wav.squeeze(0)

        return input_ids, spectrogram_labels, durations, transcript, id, wav

    def collate_fn(batch):
        input_ids = []
        spectrogram_labels = []
        duration_labels = []
        transcripts = []
        names = []
        input_values = []

        for item in batch:
            units, spectrogram, durations, transcript, id, wav = parse_item(item)
            input_ids.append(units)
            spectrogram_labels.append(spectrogram)
            duration_labels.append(durations)
            transcripts.append(transcript)
            names.append(id)
            input_values.append(wav)

        input_ids = pad_sequence(input_ids, batch_first=True)
        spectrogram_labels = pad_sequence(spectrogram_labels, batch_first=True, padding_value=-100)
        duration_labels = pad_sequence(duration_labels, batch_first=True)
        input_values = pad_sequence(input_values, batch_first=True)

        return {
            "input_ids": input_ids,
            "spectrogram_labels": spectrogram_labels,
            "duration_labels": duration_labels,
            "transcripts": transcripts,
            "names": names,
            "input_values": input_values,
        }

    return collate_fn


def resample(config):
    wav_dir_orig = Path(config.dataset.wav_dir_orig)
    wav_dir = Path(config.dataset.wav_dir)
    wav_paths = list(wav_dir_orig.glob("**/*" + config.dataset.ext_audio))

    for wav_path in tqdm(wav_paths):
        wav_name = wav_path.relative_to(wav_dir_orig)
        wav_path = str(wav_path)

        wav, sr = torchaudio.load(wav_path)
        wav = torchaudio.functional.resample(wav, sr, 16000)

        wav_path = wav_dir / wav_name
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path = str(wav_path)  # for sox backend
        torchaudio.save(wav_path, wav, 16000)


def tokenize_dataset(config):
    encoder = S5HubertForSyllableDiscovery.from_pretrained(config.speech2unit.model_name_or_path).cuda()

    trainset = LibriTTS_R(config.dataset.wav_dir, split="train-*")
    devset = LibriTTS_R(config.dataset.wav_dir, config.dataset.wav_dir_orig, split="dev-clean")
    testset = LibriTTS_R(config.dataset.wav_dir, config.dataset.wav_dir_orig, split="test-*")

    train_loader = torch.utils.data.DataLoader(trainset)
    dev_loader = torch.utils.data.DataLoader(devset)
    test_loader = torch.utils.data.DataLoader(testset)

    trainset = _tokenize(encoder, train_loader)
    devset = _tokenize(encoder, dev_loader)
    testset = _tokenize(encoder, test_loader)

    dataset = DatasetDict({"train": trainset, "dev": devset, "test": testset})
    dataset.push_to_hub(config.dataset.name)


def _tokenize(encoder, dataloader: torch.utils.data.DataLoader):
    def generate_dataset():
        for item in tqdm(dataloader):
            input_values = item["input_values"].cuda()
            input_values = input_values / input_values.abs().max() * 0.95

            spectrogram_labels = mel_spectrogram(input_values).squeeze(0)  # (80, len)
            spectrogram_labels = spectrogram_labels.transpose(0, 1)  # (len, 80)
            spectrogram_labels = spectrogram_labels.cpu().tolist()

            outputs = encoder(item["input_values"].cuda())

            item = {
                "id": item["name"][0],
                "units": outputs[0]["units"].tolist(),
                "durations": outputs[0]["durations"].tolist(),
                "transcript": item["transcript"][0],
                "spectrogram": spectrogram_labels,
            }
            yield item

    features = Features(
        {
            "id": Value("string"),
            "units": Sequence(Value("int32")),
            "durations": Sequence(Value("int32")),
            "transcript": Value("string"),
            "spectrogram": Array2D(shape=(None, 80), dtype="float32"),
        }
    )

    return Dataset.from_generator(generate_dataset, features=features)


def extract_features(config):
    wav_dir = Path(config.dataset.wav_dir)
    spectrogram_dir = Path(config.dataset.spectrogram_dir)
    wav_paths = list(wav_dir.glob("**/*" + config.dataset.ext_audio))

    for wav_path in tqdm(wav_paths):
        wav_name = wav_path.relative_to(wav_dir).with_suffix("")
        spectrogram_path = spectrogram_dir / wav_name.with_suffix(".pt")
        if spectrogram_path.is_file():
            continue
        spectrogram_path.parent.mkdir(parents=True, exist_ok=True)

        wav_path = str(wav_path)
        wav, sr = torchaudio.load(wav_path)
        wav = wav.cuda()
        wav = wav / wav.abs().max() * 0.95

        spectrogram_labels = mel_spectrogram(wav)  # (1, 80, len)
        spectrogram_labels = spectrogram_labels.transpose(1, 2)  # (1, len, 80)
        spectrogram_labels = spectrogram_labels.cpu()

        torch.save(spectrogram_labels, spectrogram_path)
