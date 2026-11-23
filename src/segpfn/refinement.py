"""Cluster smoothing and the support-evidence-constrained reference seed.

Candidate matching and seed clustering are adapted from FoRIS (Apache-2.0).
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import roc_auc_score

from .geometry import GRID, MEAN, STD


def cluster_labels(tokens):
    if len(tokens) < 2:
        return np.zeros(len(tokens), np.int64)
    distance = (1 - (tokens @ tokens.T).clamp(-1, 1)).cpu().numpy()
    return AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage="average", distance_threshold=1.0 - 0.6
    ).fit_predict(distance)


def patch_scores(probability, labels=None, guidance=None):
    """Apply optional cluster averaging, query min–max and the bounded seed residual."""
    score = np.asarray(probability, np.float32)
    if labels is not None:
        means = np.bincount(labels, weights=score) / np.bincount(labels)
        score = (0.5 * score + 0.5 * means[labels]).astype(np.float32)

    def minmax(x):
        return (x - x.min()) / max(float(np.ptp(x)), 1e-6)

    baseline = minmax(score)
    if guidance is None:
        return baseline
    seed, evidence = guidance
    delta = minmax(score + 0.25 * (seed - 0.5)) - baseline
    weight = 1 - np.clip(-np.sign(delta) * (2 * evidence - 1), 0, 1)
    return baseline + np.clip(weight * delta, -0.25, 0.25)


def _class_margin(tokens, dictionary, foreground):
    chunks = []
    for block in tokens.split(512):
        sim = block @ dictionary.T
        fg = sim[:, foreground].topk(5).values.mean(1)
        bg = sim[:, ~foreground].topk(5).values.mean(1)
        chunks.append(fg - bg)
    return torch.cat(chunks).cpu().numpy()


def evidence_score(features, targets, query_indices):
    """Calibrate foreground/background cosine margins on held-out support blocks."""
    rows = F.normalize(features.permute(0, 2, 3, 1).flatten(1, 2), dim=2)
    query = rows[-1, query_indices]
    yy, xx = np.indices((GRID, GRID))
    fold = ((yy // 8 + xx // 8) % 2).ravel()
    estimates = []
    any_valid = False
    for support, target in zip(rows[:-1], targets, strict=True):
        for held in (0, 1):
            train = np.flatnonzero((target >= 0) & (fold != held))
            test = np.flatnonzero((target >= 0) & (fold == held))
            train_y, test_y = target[train], target[test]
            neutral = np.full(len(query), 0.5, np.float32)
            if min(np.bincount(y, minlength=2).min() for y in (train_y, test_y)) < 5:
                estimates.append(neutral)
                continue
            dictionary = support[train]
            fg = torch.as_tensor(train_y == 1, device=support.device)
            held_margin = _class_margin(support[test], dictionary, fg)
            lo = np.median(held_margin[test_y == 0])
            gap = float(np.median(held_margin[test_y == 1])) - float(lo)
            reliability = max(0.0, 2 * roc_auc_score(test_y, held_margin) - 1)
            if gap <= 1e-6 or reliability == 0:
                estimates.append(neutral)
                continue
            scaled = np.clip((_class_margin(query, dictionary, fg) - lo) / gap, 0, 1)
            estimates.append((0.5 + reliability * (scaled - 0.5)).astype(np.float32))
            any_valid = True
    return np.mean(estimates, axis=0, dtype=np.float32) if any_valid else None


def reference_features(raw, debiased, masks):
    """FoRIS uses the last reference with a nearest-sampled foreground for its gate."""
    nearest = F.interpolate(masks[:, None].float(), size=(GRID, GRID), mode="nearest")[:, 0].bool()
    target_mean = raw[-1].flatten(1).mean(1)
    for feature, mask in zip(reversed(raw[:-1].unbind()), reversed(nearest.unbind()), strict=True):
        if mask.any():
            prototype = F.normalize(feature[:, mask].mean(1), dim=0)
            similarity = torch.dot(prototype, target_mean) / (target_mean.norm() + 1e-6)
            return debiased if similarity < 0.8 else raw
    return debiased


def _downsample_masks(masks):
    size = (GRID, GRID)
    small = F.interpolate(masks[:, None].float(), size=size, mode="bilinear", align_corners=False)[:, 0] > 0.5
    for i in range(len(masks)):
        if not small[i].any() and masks[i].any():
            small[i] = F.interpolate(masks[i : i + 1, None].float(), size=size, mode="nearest")[0, 0] > 0.5
            if not small[i].any():
                y, x = (masks[i].nonzero().float().mean(0) / (masks.shape[-1] // GRID)).long()
                small[i, y, x] = True
    return small


def seed_prior(features, masks, image):
    references, query = features[:-1], features[-1]
    small = _downsample_masks(masks)
    # Match on the same axes as FoRIS. Reference matching normalizes along height.
    qnorm = F.normalize(query[None], dim=1)
    votes = torch.zeros((GRID, GRID), device=query.device, dtype=torch.int32)
    prototypes = []
    for reference, mask in zip(references, small, strict=True):
        if mask.any():
            rnorm = F.normalize(reference[None], dim=2)
            similarity = torch.einsum("bchw,bcxy->bhwxy", rnorm, qnorm)[0]
            nearest = similarity.permute(2, 3, 0, 1).flatten(2).argmax(2)
            votes += mask.flatten()[nearest]
            prototypes.append(reference[:, mask].mean(1))
    if not prototypes:
        return torch.full_like(votes, 0.5, dtype=query.dtype)
    candidates = votes >= (len(references) + 1) // 2
    candidates[::2, ::2] = True
    mu = F.normalize(torch.stack(prototypes).mean(0), dim=0)

    tokens = F.normalize(query.flatten(1).T, dim=1)
    color = F.interpolate(image[None], (GRID, GRID), mode="bilinear", align_corners=False)[0]
    color = (color * color.new_tensor(STD)[:, None, None] + color.new_tensor(MEAN)[:, None, None]).clamp(0, 1)
    axis = torch.linspace(-1, 1, GRID, device=query.device)
    xy = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1).reshape(-1, 2)
    descriptors = torch.cat(
        [tokens, 0.35 * F.normalize(color.flatten(1).T, dim=1), 0.20 * F.normalize(xy, dim=1)], 1
    )
    labels = torch.as_tensor(cluster_labels(F.normalize(descriptors, dim=1)), device=query.device)
    count = int(labels.max()) + 1
    if count == 1:
        return candidates.to(query.dtype)
    similarity = torch.einsum("chw,c->hw", query, mu).flatten()
    means, cross = [], []
    for i in range(count):
        members = labels == i
        means.append(F.normalize(tokens[members].mean(0), dim=0))
        cross.append(similarity[members].mean())
    means, cross = torch.stack(means), torch.stack(cross)
    ids, hits = labels[candidates.flatten()].unique(return_counts=True)
    density = torch.zeros_like(cross)
    density[ids] = hits.to(cross.dtype)
    density /= torch.bincount(labels, minlength=count).to(cross.dtype)
    best = ids[(cross * density)[ids].argmax()]
    affinity = torch.einsum("c,kc->k", means[best], means).clamp_min(0)
    combined = cross * affinity * density
    combined[best] = combined.max().clamp_min(1e-6)
    values = (combined - combined.min()) / (combined.max() - combined.min()).clamp_min(1e-6)
    return values[labels].reshape(GRID, GRID)
