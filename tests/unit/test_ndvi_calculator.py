"""
Unit tests for src.features.ndvi_calculator module.
"""
import pytest
import numpy as np
from src.features.ndvi_calculator import (
    calculate_ndvi,
    calculate_evi,
    calculate_savi,
    classify_vegetation_stress
)


class TestCalculateNDVI:
    """Tests for NDVI calculation"""

    def test_calculate_ndvi_basic(self):
        """Test basic NDVI calculation"""
        red = np.array([[100, 200], [150, 250]])
        nir = np.array([[300, 400], [350, 450]])
        result = calculate_ndvi(red, nir)

        assert 'ndvi_masked' in result
        assert 'statistics' in result
        assert result['statistics']['mean'] > 0

    def test_calculate_ndvi_with_cloud_mask(self):
        """Test NDVI calculation with cloud mask"""
        red = np.array([[100, 200], [150, 250]])
        nir = np.array([[300, 400], [350, 450]])
        scl = np.array([[3, 0], [0, 0]])  # First pixel is cloud shadow

        result = calculate_ndvi(red, nir, scl)
        assert result['statistics']['cloud_coverage'] > 0

    def test_calculate_ndvi_zero_division(self):
        """Test NDVI handles zero division"""
        red = np.array([[0, 100]])
        nir = np.array([[0, 200]])
        result = calculate_ndvi(red, nir)
        assert not np.isnan(result['ndvi_masked']).any()


class TestCalculateEVI:
    """Tests for EVI calculation"""

    def test_calculate_evi_basic(self):
        """Test basic EVI calculation"""
        red = np.array([[100, 200]])
        nir = np.array([[300, 400]])
        blue = np.array([[50, 100]])

        evi = calculate_evi(red, nir, blue)
        assert evi.shape == red.shape
        assert not np.isnan(evi).any()


class TestCalculateSAVI:
    """Tests for SAVI calculation"""

    def test_calculate_savi_basic(self):
        """Test basic SAVI calculation"""
        red = np.array([[100, 200]])
        nir = np.array([[300, 400]])

        savi = calculate_savi(red, nir)
        assert savi.shape == red.shape
        assert not np.isnan(savi).any()

    def test_calculate_savi_different_L(self):
        """Test SAVI with different L values"""
        red = np.array([[100, 200]])
        nir = np.array([[300, 400]])

        savi_low = calculate_savi(red, nir, L=0.0)
        savi_high = calculate_savi(red, nir, L=1.0)
        assert not np.array_equal(savi_low, savi_high)


class TestClassifyVegetationStress:
    """Tests for vegetation stress classification"""

    def test_classify_vegetation_stress(self):
        """Test vegetation stress classification"""
        ndvi = np.array([[0.2, 0.4, 0.6]])

        classification = classify_vegetation_stress(ndvi)
        assert classification['high_stress'][0, 0] == True
        assert classification['medium_stress'][0, 1] == True
        assert classification['low_stress'][0, 2] == True
