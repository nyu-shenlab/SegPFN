"""Support-only PCA and the four reproducible TabPFN contexts."""

import numpy as np
import torch
from sklearn.decomposition import PCA

from .views import digest


def normalize(rows):
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)


def project(support, query, original):
    support = normalize(support)
    fit = support[original]
    rank = min(192, len(fit) - 1, fit.shape[1])
    if rank > 0:
        pca = PCA(n_components=rank, svd_solver="full").fit(fit)
        basis = np.pad(pca.components_, ((0, 192 - rank), (0, 0)))
        center = pca.mean_ @ basis.T
        # Preserve the published 192-column projection before retaining 128 columns.
        return (support @ basis.T - center)[:, :128], (normalize(query) @ basis.T - center)[:, :128]
    return np.zeros((len(support), 128), np.float32), np.zeros((len(query), 128), np.float32)


def _allocate(total, capacity):
    counts = np.minimum(total // len(capacity), capacity)
    while counts.sum() < total:
        free = np.flatnonzero(counts < capacity)
        counts[free[np.argmin(counts[free])]] += 1
    return counts


def context_rows(labels, groups, original, episode_id):
    y = labels[original]
    budget = min(len(y), 1024)
    if not budget:
        return np.empty((4, 0), np.int64)
    fg = int(y.sum())
    bg = len(y) - fg
    floor = min(64, budget // 2)
    wanted = int(np.floor(fg / len(y) * budget + 0.5))
    take_fg = int(np.clip(wanted, max(min(floor, fg), budget - bg), min(fg, budget - min(floor, bg))))
    group_ids = np.unique(groups).tolist()
    selected = [[] for _ in range(4)]
    for cls, total in enumerate((budget - take_fg, take_fg)):
        pools = [np.flatnonzero((groups == g) & (labels == cls)) for g in group_ids]
        quotas = _allocate(total, np.array([len(pool) for pool in pools]))
        for group, pool, n in zip(group_ids, pools, quotas, strict=True):
            for member, parts in enumerate(selected):
                take = pool
                if n != len(pool):
                    key = {
                        "seed": 17,
                        "episode": episode_id,
                        "rep": member,
                        "view": group,
                        "class": cls,
                        "policy": "single_aug_matched_v1",
                    }
                    rng = np.random.default_rng(int(digest(key)[:16], 16))
                    take = rng.choice(pool, n, replace=False, p=np.full(len(pool), 1 / len(pool)))
                parts.append(take)
    return np.stack([np.sort(np.concatenate(parts)) for parts in selected])


def branch_probability(classifier, support, labels, query, selected):
    classifier.inference_config = {"SUBSAMPLE_SAMPLES": list(selected)}
    classifier.fit(support, labels)
    for actual, expected in zip(
        classifier.ensemble_preprocessor_.subsample_row_indices, selected, strict=True
    ):
        if not np.array_equal(actual, expected):
            raise RuntimeError("TabPFN changed the requested support rows")
    if not np.array_equal(classifier.classes_, [0, 1]):
        raise RuntimeError("Unexpected foreground class order")
    with torch.inference_mode():
        logits = classifier.predict_raw_logits(query)
        members = [
            classifier.logits_to_probabilities(member).float().cpu().numpy()[:, 1] for member in logits
        ]
    if len(members) != 2 or not np.isfinite(members).all():
        raise RuntimeError("Expected two finite TabPFN foreground predictions")
    return np.mean(members, axis=0, dtype=np.float32)
