import hashlib
import os
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image

from segpfn import MODES
from segpfn.geometry import Geometry, decode_mask, patch_sum
from segpfn.refinement import patch_scores
from segpfn.tables import context_rows
from segpfn.views import rgb_id, support_views

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_binary_and_indexed_masks():
    for high in (1, 255):
        im = Image.fromarray(np.array([[0, high]], np.uint8))
        np.testing.assert_array_equal(decode_mask(im, (4, 1)), [[0, 0, 1, 1]])
    indexed = Image.fromarray(np.array([[0, 4, 9, 255]], np.uint8)).convert("P")
    decoded = decode_mask(indexed, indexed.size, foreground_value=4, ignore_value=255)
    np.testing.assert_array_equal(decoded, [[0, 1, 0, 255]])
    with pytest.raises(ValueError, match="class-indexed"):
        decode_mask(indexed, indexed.size)
    with pytest.raises(ValueError, match="must differ"):
        decode_mask(indexed, indexed.size, foreground_value=255, ignore_value=255)


def test_geometry_excludes_padding_and_void():
    geometry = Geometry(100, 50)
    mask = np.ones((50, 100), np.uint8)
    mask[:, :50] = 255
    foreground, labels = geometry.labels(mask)
    assert np.count_nonzero(labels == 1) == 32 * 32
    assert np.count_nonzero(labels < 0) == 64 * 64 - 32 * 32
    assert np.count_nonzero(patch_sum(geometry.valid_pixels()) >= 128) == 32 * 64
    np.testing.assert_array_equal(geometry.restore(foreground), mask == 1)


@pytest.mark.parametrize(("small", "count"), [(True, 3), (False, 5)])
def test_single_reference_views_preserve_foreground(small, count):
    with Image.open(EXAMPLES / "reference.png") as image, Image.open(EXAMPLES / "reference-mask.png") as mask:
        semantic = decode_mask(mask, image.size)
        if small:
            semantic[:] = 0
            semantic[37:40, 47:51] = 1
        views = list(support_views(image, semantic, rgb_id(image)))
        assert len(views) == count
        assert views[0][0] == 0
        for _, rgb, mapped in views:
            assert rgb.size == image.size
            assert np.count_nonzero(mapped == 1) >= np.count_nonzero(semantic)
            if small:
                np.testing.assert_array_equal(mapped, semantic)


def test_frozen_context_draws():
    labels = np.random.default_rng(21).integers(0, 2, 8000)
    groups = np.repeat([0, 1, 3, 4], 2000)
    rows = context_rows(labels, groups, groups == 0, "fixed-example")
    assert rows.shape == (4, 1024)
    assert hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest() == (
        "7fb0e4b2293a8fe8568d622d5693b811491ad760946b1b8f6be96c8a1a73d29d"
    )


def test_reference_evidence_bounds_the_residual():
    raw = np.array([0, 0.4, 0.6, 1], np.float32)
    labels = np.array([0, 0, 1, 1])
    seed = np.array([0.5, 1, 0, 0.5], np.float32)
    np.testing.assert_array_equal(patch_scores(raw), raw)
    baseline = patch_scores(raw, labels)
    assert not np.array_equal(baseline, raw)
    agrees = patch_scores(raw, labels, (seed, np.array([0.5, 1, 0, 0.5], np.float32)))
    assert agrees[1] > baseline[1] and agrees[2] < baseline[2]
    assert np.max(abs(agrees - baseline)) <= 0.25
    contradicts = patch_scores(raw, labels, (seed, np.array([0.5, 0, 1, 0.5], np.float32)))
    np.testing.assert_array_equal(contradicts, baseline)
    np.testing.assert_array_equal(patch_scores(np.ones(4, np.float32)), np.zeros(4))


@pytest.fixture(scope="module")
def gpu_model():
    from segpfn.model import SegPFN

    weights = Path(os.environ.get("SEGPFN_TEST_WEIGHTS", "weights"))
    if not torch.cuda.is_available() or not weights.is_dir():
        pytest.skip("GPU smoke tests require CUDA and downloaded checkpoints")
    return SegPFN(weights)


@pytest.mark.parametrize("mode", MODES)
def test_image_to_mask(gpu_model, mode, monkeypatch):
    from segpfn import model

    smoothing = Mock(wraps=model.cluster_labels)
    guidance = Mock(wraps=gpu_model._guidance)
    boundary = Mock(wraps=gpu_model.crf)
    monkeypatch.setattr(model, "cluster_labels", smoothing)
    monkeypatch.setattr(gpu_model, "_guidance", guidance)
    monkeypatch.setattr(gpu_model, "crf", boundary)
    with (
        Image.open(EXAMPLES / "query.png") as query,
        Image.open(EXAMPLES / "reference.png") as ref,
        Image.open(EXAMPLES / "reference-mask.png") as mask,
    ):
        options = {} if mode == "full" else {"mode": mode}
        prediction = gpu_model.predict(query, [(ref, mask)], **options)
        assert prediction.shape == (query.height, query.width)
        assert prediction.dtype == np.bool_
        assert 0 < prediction.sum() < prediction.size
        assert smoothing.call_count == int(mode != "crf-only")
        assert guidance.call_count == int(mode == "full")
        assert boundary.call_count == 1
