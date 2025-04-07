import math
from functools import partial
from multiprocessing import Pool
from typing import Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


def mincut_dp_torch(W: torch.Tensor, num_syllables: int, min_duration: int, max_duration: int):
    """
    Args:
        W (`torch.FloatTensor` of shape `(sequence_length, sequence_length)`):
            Self-similarity matrix.
    """
    T = W.shape[0]

    W_pad = torch.zeros((T + 1, T + 1), dtype=W.dtype, device=W.device)
    W_pad[1:, 1:] = W
    W = W_pad

    W_cumsum = W.cumsum(dim=0).cumsum(dim=1)  # (T + 1, T + 1)
    V = W_cumsum[-1]  # (T + 1,)

    # vol[i, j] = W[i : j + 1].sum()
    vol = V[1:][None, :] - V[:-1][:, None]  # (T, T)

    arange = torch.arange(T, device=W.device)
    i, j = torch.meshgrid(arange, arange + 1, indexing="ij")

    # W_sum[i, j] = W[i : j + 1, i : j + 1].sum()
    W_sum = W_cumsum[j, j] - 2 * W_cumsum[i, j] + W_cumsum[i, i]
    cut = vol - W_sum
    ncut = cut / (cut + W_sum / 2)

    mask = torch.tril(torch.ones_like(ncut, dtype=torch.bool), -2 + min_duration)
    ncut[mask] = float("inf")

    # gather_indices: (max_duration - min_duration + 1, T)
    gather_indices = (
        torch.as_strided(
            torch.arange(1 - max_duration, T - min_duration + 1, device=W.device),
            (T, max_duration - min_duration + 1),
            (1, 1),
        )
        .clip(0)
        .T
    )
    ncut = torch.take_along_dim(ncut, gather_indices, 0)  # (max_duration - min_duration + 1, T)

    B = torch.zeros((T + 1, num_syllables + 1), dtype=torch.int)
    C = torch.full((T + 1, num_syllables + 1), torch.finfo(W.dtype).max, device=W.device)
    C[0, 0] = 0

    # dynamic programming
    for s in range(1, num_syllables + 1):
        c = torch.take_along_dim(torch.broadcast_to(C[:T, s - 1 : s], (T, T)), gather_indices, 0) + ncut  # (d, T)
        min = torch.min(c, dim=0, keepdim=True)  # (1, T)
        B[1:, s] = torch.take_along_dim(gather_indices, min.indices, 0).squeeze(0)
        C[1:, s] = min.values.squeeze(0)

    # backtrack
    prev_b = T
    boundary = [prev_b]
    for k in range(num_syllables, 0, -1):
        prev_b = B[prev_b, k]
        boundary.append(prev_b)
    boundary.reverse()
    return boundary


def mincut_torch(
    hidden_states: torch.Tensor,
    sec_per_frame: float = 0.02,
    sec_per_syllable: float = 0.2,
    merge_threshold: Optional[float] = 0.3,
    min_duration: int = 3,
    max_duration: int = 35,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A computationally efficient PyTorch implementation of the exact minimum cut algorithm

    Args:
        hidden_states (`torch.FloatTensor` of shape `(sequence_length, hidden_size)`):
            Latent speech frame representations.
    """
    num_syllables = int(math.ceil(len(hidden_states) * sec_per_frame / sec_per_syllable))

    ssm = hidden_states @ hidden_states.T
    ssm = ssm - torch.min(ssm) + 1e-7  # make it non-negative
    seg_boundary_frame = mincut_dp_torch(ssm, num_syllables, min_duration, max_duration)

    seg_boundary_frame_pairs = [[l, r] for l, r in zip(seg_boundary_frame[:-1], seg_boundary_frame[1:])]
    pooled_feat = torch.stack([hidden_states[l:r].mean(0) for l, r in seg_boundary_frame_pairs])

    if merge_threshold is not None and len(seg_boundary_frame_pairs) >= 3:
        all_sim = torch.nn.functional.cosine_similarity(pooled_feat[:-1], pooled_feat[1:])
        min_id = torch.argmax(all_sim)
        while all_sim[min_id] >= merge_threshold and len(seg_boundary_frame_pairs) >= 3:
            l_merge, r_merge = seg_boundary_frame_pairs[min_id], seg_boundary_frame_pairs[min_id + 1]
            seg_boundary_frame_pairs = [
                pair for i, pair in enumerate(seg_boundary_frame_pairs) if i != min_id and i != min_id + 1
            ]
            seg_boundary_frame_pairs.insert(min_id, [l_merge[0], r_merge[1]])
            pooled_feat = torch.stack([hidden_states[l:r].mean(0) for l, r in seg_boundary_frame_pairs])
            all_sim = torch.nn.functional.cosine_similarity(pooled_feat[:-1], pooled_feat[1:])
            min_id = torch.argmax(all_sim)

    boundaries = torch.tensor(seg_boundary_frame_pairs, device=hidden_states.device) * sec_per_frame
    return boundaries, pooled_feat, torch.tensor(seg_boundary_frame_pairs, device=hidden_states.device)


def mincut_dp_numpy(W: np.ndarray, num_syllables: int, min_duration: int, max_duration: int):
    """
    Args:
        W (`np.ndarray` of shape `(sequence_length, sequence_length)`):
            Self-similarity matrix.
    """
    T = W.shape[0]

    W_pad = np.zeros((T + 1, T + 1), dtype=W.dtype)
    W_pad[1:, 1:] = W
    W = W_pad

    W_cumsum = W.cumsum(axis=0).cumsum(axis=1)  # (T + 1, T + 1)
    V = W_cumsum[-1]  # (T + 1,)

    # vol[i, j] = W[i : j + 1].sum()
    vol = V[1:][None, :] - V[:-1][:, None]  # (T, T)

    arange = np.arange(T)
    i, j = np.meshgrid(arange, arange + 1, indexing="ij")

    # W_sum[i, j] = W[i : j + 1, i : j + 1].sum()
    W_sum = W_cumsum[j, j] - 2 * W_cumsum[i, j] + W_cumsum[i, i]
    cut = vol - W_sum
    with np.errstate(invalid="ignore"):
        ncut = cut / (cut + W_sum / 2)

    mask = np.tril(np.ones_like(ncut, dtype=bool), -2 + min_duration)
    ncut[mask] = float("inf")

    # gather_indices: (max_duration - min_duration + 1, T)
    x = np.arange(1 - max_duration, T - min_duration + 1)
    gather_indices = (
        np.lib.stride_tricks.as_strided(x, (T, max_duration - min_duration + 1), (x.strides[0], x.strides[0])).clip(0).T
    )
    ncut = np.take_along_axis(ncut, gather_indices, 0)  # (max_duration - min_duration + 1, T)

    B = np.zeros((T + 1, num_syllables + 1), dtype=np.int32)
    C = np.full((T + 1, num_syllables + 1), np.finfo(W.dtype).max)
    C[0, 0] = 0

    # dynamic programming
    for s in range(1, num_syllables + 1):
        c = np.take_along_axis(np.broadcast_to(C[:T, s - 1 : s], (T, T)), gather_indices, 0) + ncut  # (d, T)
        min_indices = np.argmin(c, axis=0, keepdims=True)  # (1, T)
        B[1:, s] = np.take_along_axis(gather_indices, min_indices, 0).squeeze(0)
        C[1:, s] = np.take_along_axis(c, min_indices, 0).squeeze(0)

    # backtrack
    prev_b = T
    boundary = [prev_b]
    for k in range(num_syllables, 0, -1):
        prev_b = B[prev_b, k]
        boundary.append(prev_b)
    boundary.reverse()
    return boundary


def mincut_numpy(
    hidden_states: np.ndarray,
    sec_per_frame: float = 0.02,
    sec_per_syllable: float = 0.2,
    merge_threshold: Optional[float] = 0.3,
    min_duration: int = 3,
    max_duration: int = 35,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    from https://github.com/jasonppy/syllable-discovery/blob/master/save_seg_feats_mincut.py#L160
    """
    num_syllable = int(np.ceil(len(hidden_states) * sec_per_frame / sec_per_syllable))

    ssm = hidden_states @ hidden_states.T
    ssm = ssm - np.min(ssm) + 1e-7  # make it non-negative
    seg_boundary_frame = mincut_dp_numpy(ssm, num_syllable, min_duration, max_duration)

    seg_boundary_frame_pairs = [[l, r] for l, r in zip(seg_boundary_frame[:-1], seg_boundary_frame[1:])]
    pooled_feat = np.stack([hidden_states[l:r].mean(0) for l, r in seg_boundary_frame_pairs])

    if merge_threshold is not None and len(seg_boundary_frame_pairs) >= 3:
        all_sim = (
            np.sum(pooled_feat[:-1] * pooled_feat[1:], axis=1)
            / np.linalg.norm(pooled_feat[:-1], axis=1)
            / np.linalg.norm(pooled_feat[1:], axis=1)
        )
        min_id = np.argmax(all_sim)
        while all_sim[min_id] >= merge_threshold and len(seg_boundary_frame_pairs) >= 3:
            l_merge, r_merge = seg_boundary_frame_pairs[min_id], seg_boundary_frame_pairs[min_id + 1]
            seg_boundary_frame_pairs = [
                pair for i, pair in enumerate(seg_boundary_frame_pairs) if i != min_id and i != min_id + 1
            ]
            seg_boundary_frame_pairs.insert(min_id, [l_merge[0], r_merge[1]])
            pooled_feat = np.stack([hidden_states[l:r].mean(0) for l, r in seg_boundary_frame_pairs])
            all_sim = (
                np.sum(pooled_feat[:-1] * pooled_feat[1:], axis=1)
                / np.linalg.norm(pooled_feat[:-1], axis=1)
                / np.linalg.norm(pooled_feat[1:], axis=1)
            )
            min_id = np.argmax(all_sim)

    boundaries = np.array(seg_boundary_frame_pairs) * sec_per_frame
    return boundaries, pooled_feat, np.array(seg_boundary_frame_pairs)


def mincut_wrapper(
    ckpt_path,
    sec_per_frame: float = 0.02,
    sec_per_syllable: float = 0.2,
    merge_threshold: Optional[float] = 0.3,
    min_duration: int = 3,
    max_duration: int = 35,
):
    ckpt = np.load(ckpt_path, allow_pickle=True)[()]
    hidden_states = ckpt["hidden_states"]  # (n_frames, hidden_size)

    boundaries, pooled_feat, frame_boundary = mincut_numpy(
        hidden_states,
        sec_per_frame=sec_per_frame,
        sec_per_syllable=sec_per_syllable,
        merge_threshold=merge_threshold,
        min_duration=min_duration,
        max_duration=max_duration,
    )
    durations = frame_boundary[:, 1] - frame_boundary[:, 0]

    ckpt["segments"] = boundaries
    ckpt["segment_features"] = pooled_feat
    ckpt["durations"] = durations
    np.save(ckpt_path, ckpt)


def parallel_mincut(
    ckpt_paths,
    disable_tqdm: bool = True,
    sec_per_frame: float = 0.02,
    sec_per_syllable: float = 0.2,
    merge_threshold: Optional[float] = 0.3,
    min_duration: int = 3,
    max_duration: int = 35,
    num_workers: Optional[int] = None,
):
    with Pool(num_workers) as p:
        for _ in tqdm(
            p.imap_unordered(
                partial(
                    mincut_wrapper,
                    sec_per_frame=sec_per_frame,
                    sec_per_syllable=sec_per_syllable,
                    merge_threshold=merge_threshold,
                    min_duration=min_duration,
                    max_duration=max_duration,
                ),
                ckpt_paths,
            ),
            desc="minimum cut algorithm",
            total=len(ckpt_paths),
            disable=disable_tqdm,
        ):
            pass
