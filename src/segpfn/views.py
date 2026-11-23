"""Deterministic, foreground-preserving views of a single reference image."""

import hashlib
import json

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from scipy.ndimage import label

from .geometry import Geometry


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rgb_id(image):
    pixels = np.asarray(image.convert("RGB"))
    return hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()


def color_view(image, image_id, view, strength):
    rng = np.random.default_rng(
        int(digest(dict(seed=17, image=image_id, view=view, policy="paired_color_v1"))[:16], 16)
    )
    ops = {
        "brightness": ImageEnhance.Brightness,
        "contrast": ImageEnhance.Contrast,
        "saturation": ImageEnhance.Color,
    }
    amounts = {key: 1 + (rng.uniform(0.8, 1.2) - 1) * strength for key in ops}
    hue = int(round(255 * rng.uniform(-0.05, 0.05) * strength))
    for key in rng.permutation([*ops, "hue"]):
        if key == "hue":
            hsv = np.asarray(image.convert("HSV")).copy()
            hsv[..., 0] = (hsv[..., 0].astype(np.int16) + hue) % 256
            image = Image.frombytes("HSV", image.size, hsv.tobytes()).convert("RGB")
        else:
            image = ops[key](image).enhance(amounts[key])
    return image


def spatial_view(image, view, level, resample, fill):
    if view == 0:
        return ImageOps.mirror(image)
    if view in (1, 2):
        return image.rotate((15 if view == 1 else -15) * level, resample=resample, fillcolor=fill)
    dx, dy = (round(0.1 * length * level) for length in image.size)
    return image.transform(
        image.size, Image.Transform.AFFINE, (1, 0, -dx, 0, 1, -dy), resample, fillcolor=fill
    )


def fits(corners, view, level, size):
    if corners is None or view == 0:
        return True
    w, h = size
    if view == 3:
        moved = corners + [round(0.1 * w * level), round(0.1 * h * level)]
    else:
        angle = np.deg2rad((15 if view == 1 else -15) * level)
        c, s = np.cos(angle), np.sin(angle)
        center = np.array(size) / 2
        moved = (corners - center) @ np.array([[c, -s], [s, c]]) + center
    return np.all(moved >= -1e-7) and np.all(moved <= np.array(size) + 1e-7)


def support_views(image, semantic, image_id):
    """Yield the original and two weak color or four protected geometry/color views."""
    yield 0, image, semantic
    foreground = semantic == 1
    pixels = int(foreground.sum())
    fraction = pixels / max(1, np.count_nonzero(semantic != 255))
    if 0 < fraction < 0.05:
        key = digest(dict(seed=17, policy="nested_aug_count_v1", support_rgb_sha256=image_id))
        selected = sorted(np.random.default_rng(int(key[:16], 16)).permutation(4)[:2].tolist())
        for view in selected:
            yield view + 1, color_view(image, image_id, view, 0.25), semantic
        return

    geometry = Geometry(*image.size)
    original_patches = np.count_nonzero(geometry.labels(semantic)[1] == 1)
    components, count = label(foreground, structure=np.ones((3, 3)))
    component_image = Image.fromarray(components.astype(np.int32))
    mask = Image.fromarray(semantic)
    box = Image.fromarray(foreground).getbbox()
    corners = None if box is None else np.array([[x, y] for x in box[::2] for y in box[1::2]])
    for view in range(4):
        colored = color_view(image, image_id, view, 1.0)
        for level in (1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.0):
            if not fits(corners, view, level, image.size):
                continue
            mapped = np.asarray(spatial_view(mask, view, level, Image.Resampling.NEAREST, 255))
            if (
                np.count_nonzero(mapped == 1) < pixels
                or np.count_nonzero(geometry.labels(mapped)[1] == 1) < original_patches
            ):
                continue
            labels = np.asarray(spatial_view(component_image, view, level, Image.Resampling.NEAREST, 0))
            if len(np.unique(labels[labels > 0])) == count:
                rgb = spatial_view(colored, view, level, Image.Resampling.BILINEAR, (124, 116, 104))
                yield view + 1, rgb, mapped
                break
        else:
            yield view + 1, colored, semantic
