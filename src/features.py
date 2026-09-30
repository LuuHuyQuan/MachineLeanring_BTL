"""One fixed pixel representation shared by offline models and serving."""
from __future__ import annotations

import numbers

import numpy as np
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def _numeric_array(value) -> np.ndarray:
    try:
        objects = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError("Ảnh phải là mảng số hình chữ nhật.") from exc
    if not objects.size or any(isinstance(item, (bool, np.bool_)) or
                               not isinstance(item, numbers.Real) for item in objects.flat):
        raise ValueError("Pixel phải là số; không chấp nhận chuỗi, boolean hoặc null.")
    try:
        array = objects.astype(np.float64)
    except (OverflowError, ValueError, TypeError) as exc:
        raise ValueError("Pixel phải là số hữu hạn trong miền quy định.") from exc
    if not np.isfinite(array).all():
        raise ValueError("Pixel phải hữu hạn, không chứa NaN hoặc Infinity.")
    return array


def validate_pixels(value) -> np.ndarray:
    """Validate a single raw Digits image (flat64 or8x8), without clipping."""
    array = _numeric_array(value)
    if array.shape not in ((64,), (8, 8)):
        raise ValueError("Cần đúng 64 pixel hoặc ma trận 8×8.")
    if (array < 0).any() or (array > 16).any():
        raise ValueError("Miền giá trị pixel 8×8 phải nằm trong [0, 16].")
    if not np.any(array > 0):
        raise ValueError("Ảnh rỗng. Hãy vẽ một chữ số trước khi nhận dạng.")
    return array.reshape(64)


def canvas_to_pixels(value) -> np.ndarray:
    """Convert a black280x280 canvas with white strokes into raw Digits pixels.

    Crop the foreground, preserve its aspect ratio, center within an inner
    210px square, then use area averaging. No learned statistics are used.
    Both this result and original dataset images enter the same /16 pipeline.
    This deterministic adapter does not remove the canvas/domain mismatch.
    """
    array = _numeric_array(value)
    if array.shape != (280, 280):
        raise ValueError("Canvas phải là ma trận 280×280.")
    if (array < 0).any() or (array > 255).any():
        raise ValueError("Pixel canvas phải nằm trong [0, 255].")
    rows, cols = np.nonzero(array > 8)
    if len(rows) < 16:
        raise ValueError("Canvas rỗng hoặc nét vẽ quá nhỏ. Hãy vẽ rõ một chữ số.")
    crop = array[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    height, width = crop.shape
    scale = 210 / max(height, width)
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = Image.fromarray(crop.astype(np.float32)).resize(target, Image.Resampling.BILINEAR)
    centered = np.zeros((280, 280), dtype=np.float32)
    x = (280 - target[0]) // 2
    y = (280 - target[1]) // 2
    centered[y:y + target[1], x:x + target[0]] = np.asarray(resized)
    # Each8x8 cell averages one35x35 block; float values preserve grayscale.
    pixels = centered.reshape(8, 35, 8, 35).mean(axis=(1, 3)) * (16.0 / 255.0)
    return validate_pixels(pixels)


class PixelScaler(TransformerMixin, BaseEstimator):
    """Scale the known0..16 domain; never estimate bounds from held-out data."""

    @staticmethod
    def _check(X):
        array = np.asarray(X, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 64 or not np.isfinite(array).all():
            raise ValueError("Expected a finite (n_samples,64) pixel matrix.")
        if (array < 0).any() or (array > 16).any():
            raise ValueError("Pixels must be in [0,16].")
        return array

    def fit(self, X, y=None):
        self._check(X)
        self.n_features_in_ = 64
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        return self._check(X) / 16.0
