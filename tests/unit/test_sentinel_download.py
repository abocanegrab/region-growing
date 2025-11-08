"""
Unit tests for src.utils.sentinel_download module.
"""
import pytest
import numpy as np
from src.utils.sentinel_download import create_cloud_mask, create_sentinel_config


class TestCloudMask:
    """Tests for cloud mask creation"""

    def test_create_cloud_mask_with_clouds(self):
        """Test cloud mask creation with cloud pixels"""
        scl = np.array([[3, 8], [0, 4]])
        mask = create_cloud_mask(scl)
        assert mask[0, 0] == True  # Cloud shadow
        assert mask[0, 1] == True  # Cloud
        assert mask[1, 0] == False  # Clear
        assert mask[1, 1] == False  # Vegetation

    def test_create_cloud_mask_all_clear(self):
        """Test cloud mask with all clear pixels"""
        scl = np.array([[4, 5], [4, 5]])
        mask = create_cloud_mask(scl)
        assert np.sum(mask) == 0  # No clouds

    def test_create_cloud_mask_all_clouds(self):
        """Test cloud mask with all cloud pixels"""
        scl = np.array([[3, 8], [9, 10]])
        mask = create_cloud_mask(scl)
        assert np.sum(mask) == 4  # All clouds


class TestSentinelConfig:
    """Tests for Sentinel Hub configuration"""

    def test_create_sentinel_config(self):
        """Test creation of Sentinel Hub config"""
        config = create_sentinel_config("test_id", "test_secret")
        assert config.sh_client_id == "test_id"
        assert config.sh_client_secret == "test_secret"

    def test_create_sentinel_config_empty(self):
        """Test creation with empty credentials"""
        config = create_sentinel_config("", "")
        assert config.sh_client_id == ""
        assert config.sh_client_secret == ""
