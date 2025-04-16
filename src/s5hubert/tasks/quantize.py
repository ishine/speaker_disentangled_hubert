from pathlib import Path

import faiss
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def quantize(config):
    segment_dir = Path(config.path.segment_dir)
    segment_paths = []

    with open(config.dataset.train_file) as f:
        for wav_name in f:
            wav_name = wav_name.rstrip()
            segment_name = wav_name.replace(".flac", ".npy")
            segment_path = segment_dir / segment_name
            segment_paths.append(segment_path)
    segment_paths.sort()

    # 128GB CPU memory
    hidden_states = np.concatenate([np.load(path, allow_pickle=True)[()]["segment_features"] for path in segment_paths])

    Path(config.path.quantizer1).parent.mkdir(parents=True, exist_ok=True)
    Path(config.path.quantizer2).parent.mkdir(parents=True, exist_ok=True)

    quantizer1 = faiss.Kmeans(
        hidden_states.shape[1],
        config.quantizer.n_clusters1,
        niter=config.quantizer.niter,
        nredo=config.quantizer.nredo,
        verbose=config.quantizer.verbose,
        seed=config.quantizer.random_state,
        gpu=config.quantizer.gpu,
        min_points_per_centroid=config.quantizer.min_points_per_centroid,
        max_points_per_centroid=config.quantizer.max_points_per_centroid
        if config.quantizer.max_points_per_centroid
        else hidden_states.shape[0],
    )
    quantizer1.train(hidden_states)
    np.save(config.path.quantizer1, quantizer1.centroids)

    quantizer2 = AgglomerativeClustering(config.quantizer.n_clusters2)
    np.save(config.path.quantizer2, quantizer2.fit_predict(quantizer1.centroids))
