"""
Unit tests for src.utils.geo_utils module.
"""
import pytest
import numpy as np
from src.utils.geo_utils import (
    validate_bbox,
    calculate_bbox_area,
    calculate_statistics
)


class TestValidateBBox:
    """Tests for bbox validation"""

    def test_validate_bbox_valid(self):
        """Test validation with valid bbox"""
        bbox = {
            'min_lat': -12.1,
            'min_lon': -77.1,
            'max_lat': -12.0,
            'max_lon': -77.0
        }
        assert validate_bbox(bbox) == True

    def test_validate_bbox_missing_keys(self):
        """Test validation with missing keys"""
        bbox = {'min_lat': -12.1, 'min_lon': -77.1}
        assert validate_bbox(bbox) == False

    def test_validate_bbox_invalid_order(self):
        """Test validation with invalid coordinate order"""
        bbox = {
            'min_lat': -12.0,
            'min_lon': -77.0,
            'max_lat': -12.1,  # max < min
            'max_lon': -77.1
        }
        assert validate_bbox(bbox) == False

    def test_validate_bbox_out_of_range(self):
        """Test validation with out of range coordinates"""
        bbox = {
            'min_lat': -100,  # Invalid latitude
            'min_lon': -77.0,
            'max_lat': -12.0,
            'max_lon': -77.0
        }
        assert validate_bbox(bbox) == False


class TestCalculateBBoxArea:
    """Tests for bbox area calculation"""

    def test_calculate_bbox_area(self):
        """Test area calculation"""
        bbox = {
            'min_lat': -12.1,
            'min_lon': -77.1,
            'max_lat': -12.0,
            'max_lon': -77.0
        }
        area = calculate_bbox_area(bbox)
        assert area > 0
        assert isinstance(area, float)


class TestCalculateStatistics:
    """Tests for statistics calculation"""

    def test_calculate_statistics_basic(self):
        """Test basic statistics calculation"""
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        stats = calculate_statistics(values)

        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'median' in stats
        assert 'count' in stats

        assert stats['mean'] == pytest.approx(0.3, abs=0.01)
        assert stats['count'] == 5

    def test_calculate_statistics_with_mask(self):
        """Test statistics with mask"""
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        mask = np.array([False, False, True, False, False])  # Mask out 0.3

        stats = calculate_statistics(values, mask)
        assert stats['count'] == 4

    def test_calculate_statistics_empty(self):
        """Test statistics with no valid values"""
        values = np.array([0.1, 0.2, 0.3])
        mask = np.array([True, True, True])  # Mask out all

        stats = calculate_statistics(values, mask)
        assert stats['count'] == 0
        assert stats['mean'] == 0.0
