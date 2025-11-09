"""
Unit tests for Prithvi model loader.

Tests cover:
- Model loading and initialization
- Encoder forward pass
- Model information extraction
- Normalization utilities
- Interpolation utilities
- Error handling
"""
import pytest  # type: ignore
import torch  # type: ignore
from unittest.mock import Mock

from src.models.prithvi_loader import (
    PrithviEncoder,
    load_prithvi_model,
    get_model_info,
    create_simple_prithvi_model,
    normalize_hls_image,
    interpolate_embeddings,
)


class TestPrithviEncoder:
    """Tests for PrithviEncoder class."""
    
    def test_encoder_initialization(self):
        """Test encoder initialization sets model to eval mode."""
        mock_model = Mock()
        encoder = PrithviEncoder(mock_model)
        
        assert encoder.model == mock_model
        mock_model.eval.assert_called_once()
    
    def test_encoder_forward_valid_input(self):
        """Test encoder forward pass with valid input."""
        mock_model = Mock()
        mock_model.return_value = torch.randn(1, 256, 14, 14)
        
        encoder = PrithviEncoder(mock_model)
        input_tensor = torch.randn(1, 6, 224, 224)
        
        output = encoder(input_tensor)
        
        assert output.shape == (1, 256, 14, 14)
        mock_model.assert_called_once_with(input_tensor)
    
    def test_encoder_forward_invalid_ndim(self):
        """Test encoder raises error on invalid input dimensions."""
        mock_model = Mock()
        encoder = PrithviEncoder(mock_model)
        
        # 3D input (missing batch dimension)
        invalid_input = torch.randn(6, 224, 224)
        
        with pytest.raises(ValueError, match="Expected 4D input"):
            encoder(invalid_input)
    
    def test_encoder_forward_invalid_channels(self):
        """Test encoder raises error on wrong number of channels."""
        mock_model = Mock()
        encoder = PrithviEncoder(mock_model)
        
        # 3 channels instead of 6
        invalid_input = torch.randn(1, 3, 224, 224)
        
        with pytest.raises(ValueError, match="Expected 6 channels"):
            encoder(invalid_input)
    
    def test_encoder_forward_inference_failure(self):
        """Test encoder handles model inference failure."""
        mock_model = Mock()
        mock_model.side_effect = RuntimeError("Model error")
        
        encoder = PrithviEncoder(mock_model)
        input_tensor = torch.randn(1, 6, 224, 224)
        
        with pytest.raises(RuntimeError, match="Prithvi inference failed"):
            encoder(input_tensor)


class TestModelLoading:
    """Tests for load_prithvi_model function."""
    
    def test_load_simple_model_cpu(self):
        """Test loading simple model on CPU."""
        encoder = load_prithvi_model(use_simple_model=True, device='cpu')
        
        assert isinstance(encoder, PrithviEncoder)
        assert next(encoder.parameters()).device.type == 'cpu'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_load_simple_model_cuda(self):
        """Test loading simple model on CUDA."""
        encoder = load_prithvi_model(use_simple_model=True, device='cuda')
        
        assert isinstance(encoder, PrithviEncoder)
        assert next(encoder.parameters()).device.type == 'cuda'
    
    def test_load_simple_model_auto_device(self):
        """Test automatic device selection."""
        encoder = load_prithvi_model(use_simple_model=True)
        
        assert isinstance(encoder, PrithviEncoder)
        device = next(encoder.parameters()).device.type
        expected = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert device == expected
    
    def test_load_model_file_not_found(self):
        """Test error when model file not found."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_prithvi_model(
                model_path="nonexistent/model.pt",
                use_simple_model=False
            )
    
    def test_simple_model_creates_valid_architecture(self):
        """Test simple model has correct architecture."""
        encoder = load_prithvi_model(use_simple_model=True)
        
        # Test with valid input
        test_input = torch.randn(1, 6, 224, 224)
        if torch.cuda.is_available():
            test_input = test_input.cuda()
            encoder = encoder.cuda()
        
        output = encoder(test_input)
        
        # Check output shape
        assert output.ndim == 4
        assert output.shape[0] == 1
        assert output.shape[1] == 256  # Embedding dimension


class TestModelInfo:
    """Tests for get_model_info function."""
    
    def test_get_model_info_returns_dict(self):
        """Test get_model_info returns dictionary with expected keys."""
        encoder = load_prithvi_model(use_simple_model=True)
        info = get_model_info(encoder)
        
        assert isinstance(info, dict)
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
        assert 'device' in info
        assert 'dtype' in info
        assert 'in_eval_mode' in info
    
    def test_get_model_info_parameter_counts(self):
        """Test parameter counts are positive integers."""
        encoder = load_prithvi_model(use_simple_model=True)
        info = get_model_info(encoder)
        
        assert info['total_parameters'] > 0
        assert info['trainable_parameters'] > 0
        assert isinstance(info['total_parameters'], int)
        assert isinstance(info['trainable_parameters'], int)
    
    def test_get_model_info_eval_mode(self):
        """Test model is in eval mode."""
        encoder = load_prithvi_model(use_simple_model=True)
        info = get_model_info(encoder)
        
        assert info['in_eval_mode'] is True


class TestNormalization:
    """Tests for normalize_hls_image function."""
    
    def test_normalize_standardize(self):
        """Test standardization normalization."""
        image = torch.randn(1, 6, 224, 224)
        normalized = normalize_hls_image(image, method='standardize')
        
        assert normalized.shape == image.shape
        assert abs(normalized.mean().item()) < 1e-5  # Should be ~0
        assert abs(normalized.std().item() - 1.0) < 1e-5  # Should be ~1
    
    def test_normalize_minmax(self):
        """Test min-max normalization."""
        image = torch.randn(1, 6, 224, 224)
        normalized = normalize_hls_image(image, method='minmax')
        
        assert normalized.shape == image.shape
        assert normalized.min().item() >= 0.0
        assert normalized.max().item() <= 1.0
    
    def test_normalize_clip(self):
        """Test clip normalization."""
        image = torch.randn(1, 6, 224, 224)
        normalized = normalize_hls_image(image, method='clip')
        
        assert normalized.shape == image.shape
    
    def test_normalize_invalid_method(self):
        """Test error on invalid normalization method."""
        image = torch.randn(1, 6, 224, 224)
        
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_hls_image(image, method='invalid')
    
    def test_normalize_3d_input(self):
        """Test normalization works with 3D input (no batch dimension)."""
        image = torch.randn(6, 224, 224)
        normalized = normalize_hls_image(image, method='standardize')
        
        assert normalized.shape == image.shape


class TestInterpolation:
    """Tests for interpolate_embeddings function."""
    
    def test_interpolate_bilinear(self):
        """Test bilinear interpolation."""
        embeddings = torch.randn(1, 256, 14, 14)
        target_size = (224, 224)
        
        upsampled = interpolate_embeddings(embeddings, target_size, mode='bilinear')
        
        assert upsampled.shape == (1, 256, 224, 224)
    
    def test_interpolate_nearest(self):
        """Test nearest neighbor interpolation."""
        embeddings = torch.randn(1, 256, 14, 14)
        target_size = (224, 224)
        
        upsampled = interpolate_embeddings(embeddings, target_size, mode='nearest')
        
        assert upsampled.shape == (1, 256, 224, 224)
    
    def test_interpolate_batch(self):
        """Test interpolation with batch dimension."""
        embeddings = torch.randn(4, 256, 14, 14)
        target_size = (112, 112)
        
        upsampled = interpolate_embeddings(embeddings, target_size)
        
        assert upsampled.shape == (4, 256, 112, 112)
    
    def test_interpolate_downscale(self):
        """Test interpolation can downscale."""
        embeddings = torch.randn(1, 256, 224, 224)
        target_size = (56, 56)
        
        downsampled = interpolate_embeddings(embeddings, target_size)
        
        assert downsampled.shape == (1, 256, 56, 56)


class TestSimpleModel:
    """Tests for create_simple_prithvi_model function."""
    
    def test_create_simple_model_default_params(self):
        """Test creating simple model with default parameters."""
        model = create_simple_prithvi_model()
        
        assert isinstance(model, torch.nn.Module)
    
    def test_create_simple_model_custom_params(self):
        """Test creating simple model with custom parameters."""
        model = create_simple_prithvi_model(
            in_channels=6,
            embed_dim=128,
            patch_size=16,
            img_size=224
        )
        
        assert isinstance(model, torch.nn.Module)
    
    def test_simple_model_forward_pass(self):
        """Test simple model forward pass."""
        model = create_simple_prithvi_model()
        model.eval()
        
        test_input = torch.randn(1, 6, 224, 224)
        
        with torch.no_grad():
            output = model(test_input)
        
        assert output.ndim == 4
        assert output.shape[0] == 1
        assert output.shape[1] == 256
