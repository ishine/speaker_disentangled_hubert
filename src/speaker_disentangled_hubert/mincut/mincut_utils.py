from multiprocessing import Pool
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm

from . import mincut


def min_cut(
    hidden_states: np.ndarray,
    sec_per_frame: float = 0.02,
    sec_per_syllable: float = 0.2,
    merge_threshold: Optional[float] = 0.3,
    max_frames: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    from https://github.com/jasonppy/syllable-discovery/blob/master/save_seg_feats_mincut.py#L160
    """
    num_syllable = int(np.ceil(len(hidden_states) * sec_per_frame / sec_per_syllable))

    ssm = hidden_states @ hidden_states.T
    ssm = ssm - np.min(ssm) + 1e-7  # make it non-negative
    seg_boundary_frame = mincut.min_cut(ssm, num_syllable + 1, max_frames)  # +1 for the algo

    seg_boundary_frame_pairs = [[l, r] for l, r in zip(seg_boundary_frame[:-1], seg_boundary_frame[1:])]

    if merge_threshold is not None and len(seg_boundary_frame_pairs) >= 3:
        all_feat = [hidden_states[l:r].mean(0) for l, r in seg_boundary_frame_pairs]
        all_sim = [np.dot(l, r) / (np.linalg.norm(l) * np.linalg.norm(r)) for l, r in zip(all_feat[:-1], all_feat[1:])]
        min_id = np.argmax(all_sim)
        while all_sim[min_id] >= merge_threshold and len(seg_boundary_frame_pairs) >= 3:
            l_merge, r_merge = seg_boundary_frame_pairs[min_id], seg_boundary_frame_pairs[min_id + 1]
            seg_boundary_frame_pairs = [
                pair for i, pair in enumerate(seg_boundary_frame_pairs) if i != min_id and i != min_id + 1
            ]
            seg_boundary_frame_pairs.insert(min_id, [l_merge[0], r_merge[1]])
            all_feat = [hidden_states[l:r].mean(0) for l, r in seg_boundary_frame_pairs]
            all_sim = [
                np.dot(l, r) / (np.linalg.norm(l) * np.linalg.norm(r)) for l, r in zip(all_feat[:-1], all_feat[1:])
            ]
            min_id = np.argmax(all_sim)

    boundaries = np.stack([[l * sec_per_frame, r * sec_per_frame] for l, r in seg_boundary_frame_pairs])
    pooled_feat = np.stack([hidden_states[l:r].mean(0) for l, r in seg_boundary_frame_pairs])
    return boundaries, pooled_feat, np.array(seg_boundary_frame_pairs)


def mincut_wrapper(ckpt_path):
    ckpt = np.load(ckpt_path, allow_pickle=True)[()]
    hidden_states = ckpt["hidden_states"]  # (n_frames, 768)

    boundaries, pooled_feat, frame_boundary = min_cut(hidden_states)
    durations = frame_boundary[:, 1] - frame_boundary[:, 0]

    ckpt["segments"] = boundaries
    ckpt["segment_features"] = pooled_feat
    ckpt["durations"] = durations
    np.save(ckpt_path, ckpt)


def parallel_mincut(ckpt_paths, disable_tqdm: bool):
    with Pool() as p:
        for _ in tqdm(
            p.imap_unordered(mincut_wrapper, ckpt_paths),
            desc="minimum cut algorithm",
            total=len(ckpt_paths),
            disable=disable_tqdm,
        ):
            pass
