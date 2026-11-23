"""Binary few-shot segmentation from RGB images and annotated references."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits

from . import MODES
from .crf import BoundaryCRF
from .geometry import GRID, MEAN, PATCH, STD, Geometry, decode_mask, patch_sum, score_canvas
from .refinement import cluster_labels, evidence_score, patch_scores, reference_features, seed_prior
from .tables import branch_probability, context_rows, normalize, project
from .views import digest, rgb_id, support_views

CHECKPOINT = "tabpfn-v3.5-20260909.safetensors"
ENCODER_FILES = {
    "model.safetensors": "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179",
    "config.json": "135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5",
    "preprocessor_config.json": "960c41d1f3a7778b936365769a2d90550b318a6c0a53a0296957adacfe5e0dd7",
}


def _verify(path, expected):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. See the checkpoint download commands in README.md")
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            checksum.update(block)
    if checksum.hexdigest() != expected:
        raise ValueError(f"Unexpected SHA-256 for {path}. Use the revision in README.md")


@dataclass
class Encoded:
    raw: np.ndarray
    normalized: torch.Tensor
    debiased: torch.Tensor

    def rows(self, indices):
        projected = self.debiased.flatten(1).T[indices]
        return self.raw[indices], projected.contiguous().cpu().numpy()


@dataclass
class SupportView:
    rows: tuple[np.ndarray, np.ndarray]
    labels: np.ndarray
    group: int
    original: bool
    rgbxy: np.ndarray | None


@dataclass
class ReferenceGrid:
    normalized: torch.Tensor
    debiased: torch.Tensor
    foreground: np.ndarray
    labels: np.ndarray


class SegPFN:
    """Load the frozen models once and call predict for each reference/query episode."""

    def __init__(self, weights="weights"):
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        from transformers import AutoModel

        if not torch.cuda.is_available():
            raise RuntimeError("SegPFN inference requires an NVIDIA CUDA GPU")
        device = torch.device("cuda")
        torch.manual_seed(17)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.device = device
        self._classifier_lock = Lock()
        self.crf = BoundaryCRF(device)
        encoder = Path(weights) / "dinov3-large"
        for name, sha in ENCODER_FILES.items():
            _verify(encoder / name, sha)
        checkpoint = Path(weights) / CHECKPOINT
        _verify(checkpoint, "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3")
        basis_path = Path(__file__).parent / "assets" / "positional_basis.npz"
        _verify(basis_path, "6838d91c843bb59d72ccb458e58c82a9a7d705a60f8325fd97a876eae3fc8037")
        with np.load(basis_path, allow_pickle=False) as file:
            basis = torch.from_numpy(file["basis"]).to(device)
        self.projector = torch.eye(basis.shape[0], device=device) - basis @ basis.T
        self.encoder = (
            AutoModel.from_pretrained(encoder, local_files_only=True)
            .float()
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        self.prefix = 1 + self.encoder.config.num_register_tokens
        self.classifier = TabPFNClassifier.create_default_for_version(
            ModelVersion.V3_5,
            model_path=checkpoint,
            device=str(device),
            random_state=17,
            n_estimators=2,
            n_preprocessing_jobs=1,
            inference_precision=torch.float32,
            fit_mode="fit_preprocessors",
            show_progress_bar=False,
            categorical_features_indices=[],
        )

    @torch.inference_mode()
    def _encode(self, canvas):
        rgb = torch.from_numpy(((canvas - MEAN) / STD).transpose(2, 0, 1).copy()).to(self.device)
        raw = self.encoder(pixel_values=rgb[None]).last_hidden_state[0, self.prefix :].contiguous()
        feature_map = raw.T.contiguous().reshape(1, 1, -1, GRID, GRID)
        normalized = F.normalize(feature_map, dim=2)
        debiased = (self.projector[None] @ normalized.flatten(3)[0]).reshape_as(normalized)
        return Encoded(raw.cpu().numpy(), normalized[0, 0], F.normalize(debiased, dim=2)[0, 0]), rgb

    def _support(self, references, mode, foreground_value, ignore_value):
        views, grids, ids = [], [], []
        multi = len(references) > 1
        for i, (image, mask) in enumerate(references):
            image = image.convert("RGB")
            semantic = decode_mask(mask, image.size, foreground_value, ignore_value)
            image_id = rgb_id(image)
            ids.append((image_id, hashlib.sha256(semantic.tobytes()).hexdigest()))
            geometry = Geometry(*image.size)
            augmented = [(0, image, semantic)] if multi else support_views(image, semantic, image_id)
            for view_id, rgb, view_mask in augmented:
                foreground, labels = geometry.labels(view_mask)
                keep = np.flatnonzero(labels >= 0)
                canvas = geometry.canvas(rgb)
                feature, _ = self._encode(canvas)
                extra = geometry.rgbxy(canvas)[keep] if multi else None
                views.append(
                    SupportView(
                        feature.rows(keep), labels[keep], i if multi else view_id, view_id == 0, extra
                    )
                )
                if view_id == 0 and mode == "full":
                    grids.append(ReferenceGrid(feature.normalized, feature.debiased, foreground, labels))
        return views, grids, ids

    def _predict_patches(self, views, query_rows, query_rgbxy, episode_id):
        labels = np.concatenate([view.labels for view in views])
        original = np.repeat([view.original for view in views], [len(view.labels) for view in views])
        groups = np.repeat([view.group for view in views], [len(view.labels) for view in views])
        selected = context_rows(labels, groups, original, episode_id)
        extra = np.concatenate([view.rgbxy for view in views]) if query_rgbxy is not None else None
        branches = []
        # The reusable classifier is fitted on each branch's episode context.
        with self._classifier_lock, threadpool_limits(limits=4, user_api="blas"):
            for j in range(2):
                rows = np.concatenate([view.rows[j] for view in views])
                xs, xq = project(rows, query_rows[j], original)
                if extra is not None:
                    xs, xq = np.column_stack([xs, extra]), np.column_stack([xq, query_rgbxy])
                torch.manual_seed(17)
                members = selected[2 * j : 2 * j + 2]
                branches.append(
                    branch_probability(
                        self.classifier, np.ascontiguousarray(xs), labels, np.ascontiguousarray(xq), members
                    )
                )
        return (branches[0] + branches[1]) * np.float32(0.5)

    @torch.inference_mode()
    def _guidance(self, grids, query, image, indices):
        raw = torch.stack([g.normalized for g in grids] + [query.normalized])
        debiased = torch.stack([g.debiased for g in grids] + [query.debiased])
        masks = torch.from_numpy(np.stack([g.foreground for g in grids])).to(self.device)
        features = reference_features(raw, debiased, masks)
        evidence = evidence_score(features, [g.labels for g in grids], indices)
        if evidence is None:
            return None
        prior = seed_prior(features, masks, image).flatten()[indices].cpu().numpy()
        return prior, evidence

    def predict(
        self, query, references, *, mode="full", episode_id=None, foreground_value=None, ignore_value=None
    ):
        """Return a boolean (height, width) mask. Inputs are PIL images, references are (image, mask) pairs."""
        if mode not in MODES:
            raise ValueError(f"Choose one mode from {MODES}")
        references = list(references)
        if not references:
            raise ValueError("Provide at least one reference image and mask")
        geometry = Geometry(*query.size)
        indices = np.flatnonzero(patch_sum(geometry.valid_pixels()) >= PATCH**2 // 2)
        views, grids, ids = self._support(references, mode, foreground_value, ignore_value)
        original_labels = np.concatenate([view.labels for view in views if view.original])
        if not len(indices) or len(np.unique(original_labels)) < 2:
            # A constant PFN map gives zero after min–max and has no usable evidence folds.
            return np.zeros((query.height, query.width), bool)
        if episode_id is None:
            episode_id = digest({"query": rgb_id(query), "references": ids})
        canvas = geometry.canvas(query)
        features, rgb = self._encode(canvas)
        rows = features.rows(indices)
        extra = geometry.rgbxy(canvas)[indices] if len(references) > 1 else None
        probability = self._predict_patches(views, rows, extra, episode_id)
        with torch.inference_mode():
            clusters = None
            if mode != "crf-only":
                clusters = cluster_labels(torch.from_numpy(normalize(rows[0])).to(self.device))
            guidance = self._guidance(grids, features, rgb, indices) if mode == "full" else None
            score = patch_scores(probability, clusters, guidance)
            return geometry.restore(self.crf(rgb, score_canvas(score, indices) >= 0.5))
