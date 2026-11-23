"""The shared 1024-pixel encoder, support-label and output geometry."""

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

SIZE = 1024
PATCH = 16
GRID = SIZE // PATCH
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass(frozen=True)
class Geometry:
    width: int
    height: int

    @property
    def box(self):
        scale = SIZE / max(self.width, self.height)
        w, h = (max(1, int(v * scale + 0.5)) for v in (self.width, self.height))
        x, y = (SIZE - w) // 2, (SIZE - h) // 2
        return x, y, x + w, y + h

    def canvas(self, image):
        x, y, right, bottom = self.box
        resized = image.convert("RGB").resize((right - x, bottom - y), Image.Resampling.BILINEAR)
        canvas = np.broadcast_to(MEAN, (SIZE, SIZE, 3)).copy()
        canvas[y:bottom, x:right] = np.asarray(resized, np.float32) / 255
        return canvas

    def labels(self, semantic):
        x, y, right, bottom = self.box
        resized = Image.fromarray(semantic).resize((right - x, bottom - y), Image.Resampling.NEAREST)
        canvas = np.full((SIZE, SIZE), 255, np.uint8)
        canvas[y:bottom, x:right] = np.asarray(resized)
        foreground = canvas == 1
        valid = patch_sum(canvas != 255)
        labels = np.where(valid >= PATCH**2 // 2, 2 * patch_sum(foreground) >= valid, -1)
        return foreground, labels.ravel()

    def valid_pixels(self):
        x, y, right, bottom = self.box
        valid = np.zeros((SIZE, SIZE), bool)
        valid[y:bottom, x:right] = True
        return valid

    def rgbxy(self, canvas):
        x, y, right, bottom = self.box
        valid = self.valid_pixels()
        count = np.maximum(patch_sum(valid).ravel(), 1)
        # Average pixel-center coordinates and colors over the unpadded patch.
        yy, xx = np.ogrid[:SIZE, :SIZE]
        channels = [*canvas.transpose(2, 0, 1), (xx + 0.5 - x) / (right - x), (yy + 0.5 - y) / (bottom - y)]
        return np.stack([patch_sum(channel * valid).ravel() / count for channel in channels], axis=1).astype(
            np.float32
        )

    def restore(self, mask):
        cropped = Image.fromarray(np.asarray(mask, np.float32)).crop(self.box)
        return np.asarray(cropped.resize((self.width, self.height), Image.Resampling.BILINEAR)) >= 0.5


def patch_sum(pixels):
    dtype = np.float64 if np.issubdtype(pixels.dtype, np.floating) else np.int64
    return pixels.reshape(GRID, PATCH, GRID, PATCH).sum(axis=(1, 3), dtype=dtype)


def score_canvas(scores, indices):
    grid = np.full(GRID**2, np.nan, np.float32)
    grid[indices] = scores
    grid = grid.reshape(GRID, GRID)
    missing = np.isnan(grid)
    if missing.any():
        nearest = distance_transform_edt(missing, return_distances=False, return_indices=True)
        grid[missing] = grid[tuple(nearest[:, missing])]
    return np.asarray(Image.fromarray(grid).resize((SIZE, SIZE), Image.Resampling.BILINEAR))


def decode_mask(mask, size, foreground_value=None, ignore_value=None):
    """Map binary or indexed reference masks to background=0, foreground=1, void=255."""
    values = np.asarray(mask)
    if values.ndim != 2 or values.dtype.kind not in "uib":
        raise ValueError("Use a single-channel integer or indexed reference mask")
    if foreground_value is None:
        present = set(np.unique(values).tolist()) - {0, ignore_value}
        if present not in (set(), {1}, {255}):
            raise ValueError(
                "For a class-indexed mask, specify --foreground-value and optionally --ignore-value"
            )
        foreground_value = next(iter(present), None)
    if foreground_value is not None and foreground_value == ignore_value:
        raise ValueError("Foreground and ignore values must differ")
    semantic = np.zeros(values.shape, np.uint8)
    if foreground_value is not None:
        semantic[values == foreground_value] = 1
    if ignore_value is not None:
        semantic[values == ignore_value] = 255
    if mask.size != size:
        semantic = np.asarray(Image.fromarray(semantic).resize(size, Image.Resampling.NEAREST))
    return semantic
