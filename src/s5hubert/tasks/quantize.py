from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans


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

    hidden_states = np.concatenate([np.load(path, allow_pickle=True)[()]["segment_features"] for path in segment_paths])

    Path(config.path.quantizer1).parent.mkdir(parents=True, exist_ok=True)
    Path(config.path.quantizer2).parent.mkdir(parents=True, exist_ok=True)

    quantizer1 = MiniBatchKMeans(
        n_clusters=config.quantizer.n_clusters1,
        batch_size=config.quantizer.batch_size,
        verbose=config.quantizer.verbose,
        compute_labels=config.quantizer.compute_labels,
        random_state=config.quantizer.random_state,
        max_no_improvement=config.quantizer.max_no_improvement,
        n_init=config.quantizer.n_init,
        reassignment_ratio=config.quantizer.reassignment_ratio,
    )
    quantizer1.fit(hidden_states)
    joblib.dump(quantizer1, config.path.quantizer1)

    quantizer2 = AgglomerativeClustering(config.quantizer.n_clusters2)
    np.save(config.path.quantizer2, quantizer2.fit_predict(quantizer1.cluster_centers_))
