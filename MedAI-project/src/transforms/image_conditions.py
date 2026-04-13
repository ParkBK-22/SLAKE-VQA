from __future__ import annotations

from typing import Callable

import numpy as np
from PIL import Image, ImageFilter


ConditionFn = Callable[..., Image.Image]


def _to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def apply_original(image: Image.Image) -> Image.Image:
    return _to_rgb(image)


def apply_black(image: Image.Image) -> Image.Image:
    image = _to_rgb(image)
    arr = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def apply_lpf(image: Image.Image, sigma: float = 3.0) -> Image.Image:
    image = _to_rgb(image)
    return image.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def apply_hpf(image: Image.Image, sigma: float = 3.0) -> Image.Image:
    image = _to_rgb(image)
    original = np.asarray(image).astype(np.float32)
    blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=float(sigma)))).astype(np.float32)
    high = original - blurred
    high = high - high.min()
    if high.max() > 0:
        high = high / high.max() * 255.0
    high = high.clip(0, 255).astype(np.uint8)
    return Image.fromarray(high)


def apply_patch_shuffle(image: Image.Image, patch_size: int = 16, seed: int = 42) -> Image.Image:
    image = _to_rgb(image)
    arr = np.asarray(image)
    h, w, c = arr.shape
    ps = int(patch_size)

    patches = []
    coords = []
    for y in range(0, h, ps):
        for x in range(0, w, ps):
            patch = arr[y:min(y + ps, h), x:min(x + ps, w), :].copy()
            patches.append(patch)
            coords.append((y, x))

    rng = np.random.default_rng(seed)
    rng.shuffle(patches)

    out = np.zeros_like(arr)
    for (y, x), patch in zip(coords, patches):
        ph, pw = patch.shape[:2]
        out[y:y + ph, x:x + pw, :] = patch

    return Image.fromarray(out)


def get_condition_fn(name: str) -> ConditionFn:
    name = name.lower()
    if name == "original":
        return apply_original
    if name == "black":
        return apply_black
    if name == "lpf":
        return apply_lpf
    if name == "hpf":
        return apply_hpf
    if name == "patch_shuffle":
        return apply_patch_shuffle
    raise ValueError(f"Unknown condition: {name}")