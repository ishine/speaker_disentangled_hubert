import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torchaudio

from .nansy import _load_waveform, change_gender, random_eq


class LibriSpeech(torchaudio.datasets.LIBRISPEECH):
    def __init__(
        self,
        root: Union[str, Path] = "data",
        url: str = "train-clean-100",
        folder_in_archive: str = "LibriSpeech",
        download: bool = False,
        max_sample_size: Optional[int] = 80080,
    ):
        super().__init__(root, url, folder_in_archive, download)
        self.max_sample_size = max_sample_size

    def __getitem__(self, n: int) -> Dict[str, Any]:
        metadata = self.get_metadata(n)
        teacher_input_values = _load_waveform(self._archive, metadata[0], metadata[1])

        if self.max_sample_size is not None:
            diff = len(teacher_input_values) - self.max_sample_size
            if diff > 0:
                start = random.randrange(diff)
                teacher_input_values = teacher_input_values[start : start + self.max_sample_size]

        student_input_values = self.perturb_waveform(teacher_input_values, metadata[1])

        return {
            "teacher_input_values": torch.from_numpy(teacher_input_values),
            "student_input_values": torch.from_numpy(student_input_values),
            "spk_id": metadata[3],
        }

    def perturb_waveform(self, waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
        student_input_values = change_gender(waveform, sr)
        student_input_values = random_eq(student_input_values, sr)
        return np.clip(student_input_values, -1.0, 1.0)

    @staticmethod
    def collate_fn(batch) -> Dict[str, torch.Tensor]:
        teacher_lengths = [len(item["teacher_input_values"]) for item in batch]
        student_lengths = [len(item["student_input_values"]) for item in batch]

        bsz = len(batch)
        max_teacher_len = max(teacher_lengths)
        max_student_len = max(student_lengths)

        teacher_input_values = torch.zeros(bsz, max_teacher_len)
        student_input_values = torch.zeros(bsz, max_student_len)
        teacher_attention_mask = torch.ones(bsz, max_teacher_len, dtype=torch.long)
        student_attention_mask = torch.ones(bsz, max_student_len, dtype=torch.long)

        for n, item in enumerate(batch):
            teacher_input_values[n, : teacher_lengths[n]] = item["teacher_input_values"]
            student_input_values[n, : student_lengths[n]] = item["student_input_values"]
            teacher_attention_mask[n, teacher_lengths[n] :] = 0
            student_attention_mask[n, student_lengths[n] :] = 0

        return {
            "teacher_input_values": teacher_input_values,
            "student_input_values": student_input_values,
            "teacher_attention_mask": teacher_attention_mask,
            "student_attention_mask": student_attention_mask,
        }


class VoxCeleb(torchaudio.datasets.VoxCeleb1Identification):
    def __init__(
        self,
        root: Union[str, Path] = "data/VoxCeleb1",
        subset: str = "train",
        meta_url: str = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt",
        download: bool = False,
        max_sample_size: Optional[int] = 128000,
    ):
        super().__init__(root, subset, meta_url, download)
        self.max_sample_size = max_sample_size

    def __getitem__(self, n: int) -> Dict[str, Any]:
        metadata = self.get_metadata(n)
        waveform = _load_waveform(self._path, metadata[0], metadata[1])

        if self.max_sample_size is not None:
            attention_mask = torch.ones(self.max_sample_size, dtype=torch.long)
            diff = len(waveform) - self.max_sample_size
            if diff > 0:
                start = random.randrange(diff)
                waveform = waveform[start : start + self.max_sample_size]
            else:  # need to pad
                waveform = np.concatenate([waveform, np.zeros(-diff, dtype=waveform.dtype)])
                attention_mask[diff:] = 0
        else:
            attention_mask = torch.ones(len(waveform), dtype=torch.long)

        return {
            "waveform": torch.from_numpy(waveform),
            "attention_mask": attention_mask,
            "labels": metadata[2] - 1,
        }
