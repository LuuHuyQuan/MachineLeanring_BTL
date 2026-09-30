"""Meaningful input and fixed-scaling checks, independent of fitted models."""

import numpy as np
import pytest
from sklearn.base import clone

from src.features import PixelScaler, canvas_to_pixels, validate_pixels


def valid_pixels():
    image = np.zeros((8, 8), dtype=float)
    image[1:7, 3:5] = 16
    return image


def test_pixels_flat_and_matrix_have_the_same_row_order():
    image = valid_pixels()
    image[2, 4] = 7.5
    from_matrix = np.asarray(validate_pixels(image.tolist()))
    from_flat = np.asarray(validate_pixels(image.reshape(-1).tolist()))
    assert from_matrix.shape == (64,)
    np.testing.assert_array_equal(from_matrix, from_flat)
    assert from_matrix[2 * 8 + 4] == 7.5


@pytest.mark.parametrize("bad", [[], [1] * 63, [1] * 65, [[1] * 8] * 7])
def test_pixels_reject_empty_or_wrong_shape(bad):
    with pytest.raises(ValueError):
        validate_pixels(bad)


@pytest.mark.parametrize("bad_value", [-0.01, 16.01, np.nan, np.inf, -np.inf, True, "8", None])
def test_pixels_reject_invalid_element_without_silent_conversion(bad_value):
    image = valid_pixels().reshape(-1).tolist()
    image[0] = bad_value
    with pytest.raises(ValueError):
        validate_pixels(image)


def test_blank_pixels_rejected_for_prediction():
    with pytest.raises(ValueError):
        validate_pixels(np.zeros((8, 8)).tolist())


def test_scaler_uses_known_range_and_does_not_learn_train_maximum():
    train = np.full((2, 64), 8.0)
    unseen = np.full((1, 64), 16.0)
    scaler = PixelScaler().fit(train)
    np.testing.assert_allclose(scaler.transform(train), 0.5)
    np.testing.assert_allclose(scaler.transform(unseen), 1.0)
    np.testing.assert_allclose(clone(scaler).fit(train).transform(unseen), 1.0)


def test_scaler_keeps_zero_background_and_does_not_mutate_input():
    pixels = np.vstack([np.zeros(64), np.full(64, 16.0)])
    before = pixels.copy()
    output = PixelScaler().fit_transform(pixels)
    np.testing.assert_array_equal(pixels, before)
    np.testing.assert_array_equal(output[0], np.zeros(64))
    np.testing.assert_array_equal(output[1], np.ones(64))


def test_canvas_white_ink_becomes_finite_digits_features():
    canvas = np.zeros((280, 280), dtype=float)
    canvas[45:235, 115:165] = 255
    before = canvas.copy()
    pixels = np.asarray(canvas_to_pixels(canvas.tolist()))
    assert pixels.shape == (64,)
    assert np.isfinite(pixels).all()
    assert 0 <= pixels.min() <= pixels.max() <= 16
    assert np.count_nonzero(pixels) > 0
    assert pixels.min() < pixels.max()
    np.testing.assert_array_equal(canvas, before)


@pytest.mark.parametrize("bad", [[], [[255] * 280] * 279, [[255] * 279] * 280, np.zeros((280, 280)).tolist()])
def test_canvas_rejects_wrong_shape_or_no_ink(bad):
    with pytest.raises(ValueError):
        canvas_to_pixels(bad)


@pytest.mark.parametrize("bad_value", [-1, 256, np.nan, np.inf, True, "255", None])
def test_canvas_rejects_non_numeric_non_finite_and_out_of_range(bad_value):
    canvas = np.zeros((280, 280), dtype=float).tolist()
    canvas[80][100] = 255
    canvas[0][0] = bad_value
    with pytest.raises(ValueError):
        canvas_to_pixels(canvas)
