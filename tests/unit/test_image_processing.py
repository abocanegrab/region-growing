"""
Unit tests for src.utils.image_processing module.
"""
import pytest
import numpy as np
from src.utils.image_processing import (
    normalize_band,
    create_rgb_image,
    create_false_color_image
)


class TestNormalizeBand:
    """Tests for band normalization"""

    def test_normalize_band_percentile(self):
        """Test percentile normalization"""
        band = np.array([[0, 5000], [10000, 15000]])
        normalized = normalize_band(band, method='percentile')
        assert normalized.min() >= 0
        assert normalized.max() <= 1

    def test_normalize_band_minmax(self):
        """Test minmax normalization"""
        band = np.array([[100, 200], [300, 400]])
        normalized = normalize_band(band, method='minmax')
        assert np.isclose(normalized.min(), 0.0)
        assert np.isclose(normalized.max(), 1.0)

    def test_normalize_band_invalid_method(self):
        """Test invalid normalization method"""
        band = np.array([[100, 200]])
        with pytest.raises(ValueError):
            normalize_band(band, method='invalid')


class TestCreateRGBImage:
    """Tests for RGB image creation"""

    def test_create_rgb_image_shape(self):
        """Test RGB image has correct shape"""
        red = np.ones((10, 10)) * 100
        green = np.ones((10, 10)) * 150
        blue = np.ones((10, 10)) * 200
        rgb = create_rgb_image(red, green, blue)
        assert rgb.shape == (10, 10, 3)
        assert rgb.dtype == np.uint8

    def test_create_rgb_image_range(self):
        """Test RGB values are in valid range"""
        red = np.random.rand(10, 10) * 10000
        green = np.random.rand(10, 10) * 10000
        blue = np.random.rand(10, 10) * 10000
        rgb = create_rgb_image(red, green, blue)
        assert rgb.min() >= 0
        assert rgb.max() <= 255


class TestCreateFalseColorImage:
    """Tests for false color image creation"""

    def test_create_false_color_image_shape(self):
        """Test false color image has correct shape"""
        nir = np.ones((10, 10)) * 300
        red = np.ones((10, 10)) * 150
        green = np.ones((10, 10)) * 100
        fc = create_false_color_image(nir, red, green)
        assert fc.shape == (10, 10, 3)
        assert fc.dtype == np.uint8

    def test_create_false_color_image_range(self):
        """Test false color values are in valid range"""
        nir = np.random.rand(10, 10) * 10000
        red = np.random.rand(10, 10) * 10000
        green = np.random.rand(10, 10) * 10000
        fc = create_false_color_image(nir, red, green)
        assert fc.min() >= 0
        assert fc.max() <= 255
