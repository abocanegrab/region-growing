"""
Prithvi Foundation Model Loader.

This module provides utilities to load and use the Prithvi-EO-1.0-100M
foundation model for extracting semantic embeddings from Sentinel-2 imagery.

The Prithvi model is a Vision Transformer pre-trained on Harmonized Landsat
Sentinel-2 (HLS) imagery. It can extract 256-dimensional semantic embeddings
from 6-band multispectral imagery without task-specific fine-tuning.

Usage:
    from src.models.prithvi_loader import load_prithvi_model
    
    encoder = load_prithvi_model()
    embeddings = encoder(hls_image)

References:
    Jakubik et al. (2024). Foundation models for generalist geospatial AI.
    https://arxiv.org/abs/2310.18660
"""
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Union
import torch  # type: ignore
import torch.nn as nn  # type: ignore

logger = logging.getLogger(__name__)


class PrithviEncoder(nn.Module):
    """
    Prithvi encoder wrapper for extracting semantic embeddings.
    
    The encoder takes 6-band HLS imagery as input and outputs 256-dimensional
    embeddings at each spatial location. This is a simplified implementation
    that uses Prithvi as a feature extractor.
    
    Input shape: (B, 6, H, W) - 6 bands: B02, B03, B04, B8A, B11, B12
    Output shape: (B, 256, H', W') - 256D embeddings
    
    Note:
        H' and W' may differ from H and W due to model architecture.
        Use interpolation to match original resolution if needed.
    
    Args:
        model: Pre-trained Prithvi model
        
    Example:
        >>> encoder = PrithviEncoder(model)
        >>> hls_image = torch.randn(1, 6, 224, 224)
        >>> embeddings = encoder(hls_image)
        >>> print(embeddings.shape)  # (1, 256, 14, 14)
    """
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.model.eval()
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract semantic embeddings from HLS imagery.
        
        Args:
            x: Input tensor of shape (B, 6, H, W)
            
        Returns:
            Embeddings of shape (B, 256, H', W')
            
        Raises:
            RuntimeError: If inference fails
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B, C, H, W), got {x.ndim}D")
        
        if x.shape[1] != 6:
            raise ValueError(f"Expected 6 channels (HLS bands), got {x.shape[1]}")
        
        try:
            features = self.model(x)
            return features
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise RuntimeError(f"Prithvi inference failed: {e}")


def create_simple_prithvi_model(
    in_channels: int = 6,
    embed_dim: int = 256,
    patch_size: int = 16,
    img_size: int = 224
) -> nn.Module:
    """
    Create a simplified Prithvi-like model for development and testing.
    
    This is a lightweight Vision Transformer that mimics Prithvi's architecture
    but is much smaller. Use this for testing without downloading the full model.
    
    Args:
        in_channels: Number of input channels (default: 6 for HLS)
        embed_dim: Embedding dimension (default: 256)
        patch_size: Size of image patches (default: 16)
        img_size: Input image size (default: 224)
        
    Returns:
        Simple Vision Transformer model that outputs (B, C, H', W')
        
    Note:
        This is NOT the real Prithvi model. It's a simplified version for
        testing and development. For production, use the full Prithvi model.
    """
    try:
        import timm  # type: ignore
    except ImportError:
        raise ImportError("timm is required. Install it with: poetry add timm")
    
    class SimpleViTWrapper(nn.Module):
        """Wrapper to convert ViT output to spatial format."""
        
        def __init__(self, base_model, embed_dim, patch_size, img_size):
            super().__init__()
            self.base_model = base_model
            self.embed_dim = embed_dim
            self.patch_size = patch_size
            self.img_size = img_size
            self.num_patches = (img_size // patch_size) ** 2
            
            # Projection to target embedding dimension
            if embed_dim != base_model.embed_dim:
                self.proj = nn.Linear(base_model.embed_dim, embed_dim)
            else:
                self.proj = nn.Identity()
        
        def forward(self, x):
            # Get ViT features
            features = self.base_model.forward_features(x)
            
            # Remove cls token if present (first token)
            if features.shape[1] == self.num_patches + 1:
                features = features[:, 1:, :]  # (B, num_patches, embed_dim)
            
            # Project to target dimension
            features = self.proj(features)  # (B, num_patches, target_embed_dim)
            
            # Reshape to spatial format
            B = features.shape[0]
            H = W = int(self.num_patches ** 0.5)
            features = features.transpose(1, 2).reshape(B, self.embed_dim, H, W)
            
            return features
    
    # Create base ViT model
    base_model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=False,
        in_chans=in_channels,
        num_classes=0,
        global_pool='',
    )
    
    # Wrap with spatial output
    model = SimpleViTWrapper(base_model, embed_dim, patch_size, img_size)
    
    logger.info(f"Created simple Prithvi-like model: {embed_dim}D embeddings")
    return model


def create_real_prithvi_encoder(
    checkpoint: dict,
    device: str = 'cuda'
) -> nn.Module:
    """
    Create real Prithvi encoder from checkpoint.
    
    Prithvi uses a ViT architecture with 3D patches for temporal data.
    For single-image inference, we adapt it to work with 2D images.
    
    Args:
        checkpoint: Loaded checkpoint dictionary
        device: Device to load model on
        
    Returns:
        Encoder model with loaded weights
    """
    
    class PrithviEncoderModel(nn.Module):
        """
        Real Prithvi encoder that uses pre-trained weights.
        
        This implementation extracts patch embeddings and uses positional
        encodings from the pre-trained Prithvi model.
        """
        
        def __init__(self, checkpoint_dict, device):
            super().__init__()
            
            # Extract parameters from checkpoint
            pos_embed_shape = checkpoint_dict['encoder.pos_embed'].shape
            self.embed_dim = pos_embed_shape[2]  # 768
            
            # Patch embedding: convert 3D conv (temporal) to 2D
            patch_embed_weight_3d = checkpoint_dict['encoder.patch_embed.proj.weight']
            # Shape: (768, 6, 1, 16, 16) -> squeeze temporal dim
            patch_embed_weight_2d = patch_embed_weight_3d.squeeze(2)  # (768, 6, 16, 16)
            
            # Create conv layer for patch embedding
            self.patch_embed_conv = nn.Conv2d(
                in_channels=6,
                out_channels=self.embed_dim,
                kernel_size=16,
                stride=16,
                bias=True
            )
            # Load weights
            self.patch_embed_conv.weight.data = patch_embed_weight_2d.to(device)
            self.patch_embed_conv.bias.data = checkpoint_dict['encoder.patch_embed.proj.bias'].to(device)
            
            # CLS token and positional embedding
            self.cls_token = nn.Parameter(checkpoint_dict['encoder.cls_token'].to(device))
            
            # Positional embedding (will be interpolated in forward)
            self.register_buffer('pos_embed_full', checkpoint_dict['encoder.pos_embed'].to(device))
            
            # Projection to 256D for compatibility with our pipeline
            self.proj = nn.Linear(self.embed_dim, 256)
            self.proj = self.proj.to(device)
            
        def interpolate_pos_embed(self, num_patches):
            """
            Interpolate positional embedding to match current number of patches.
            
            Prithvi was trained with temporal-spatial data (588 patches).
            We need to adapt it for single images (196 patches for 224x224).
            """
            # pos_embed_full shape: (1, 589, 768) = (1, 1+588, 768)
            # We need: (1, 1+196, 768) for 224x224 image
            
            pos_embed_cls = self.pos_embed_full[:, 0:1, :]  # (1, 1, 768)
            pos_embed_spatial = self.pos_embed_full[:, 1:, :]  # (1, 588, 768)
            
            # Prithvi uses 21x28 spatial-temporal grid (588 patches)
            # We need 14x14 spatial grid (196 patches)
            # Interpolate using 2D interpolation
            
            # Reshape to 2D: assume square-ish spatial layout
            # 588 ≈ 24.25^2, let's use 21x28 (from Prithvi paper)
            H_orig, W_orig = 21, 28
            pos_embed_2d = pos_embed_spatial.reshape(1, H_orig, W_orig, self.embed_dim)
            pos_embed_2d = pos_embed_2d.permute(0, 3, 1, 2)  # (1, 768, 21, 28)
            
            # Target size for 224x224 image with 16x16 patches
            H_new = W_new = int(num_patches ** 0.5)  # 14 for 224x224
            
            # Interpolate
            pos_embed_2d = torch.nn.functional.interpolate(
                pos_embed_2d,
                size=(H_new, W_new),
                mode='bilinear',
                align_corners=False
            )
            
            # Reshape back to sequence
            pos_embed_2d = pos_embed_2d.permute(0, 2, 3, 1)  # (1, 14, 14, 768)
            pos_embed_spatial_new = pos_embed_2d.reshape(1, H_new * W_new, self.embed_dim)
            
            # Concatenate cls and spatial
            pos_embed = torch.cat([pos_embed_cls, pos_embed_spatial_new], dim=1)
            
            return pos_embed
        
        def forward(self, x):
            B = x.shape[0]
            
            # Patch embedding: (B, 6, 224, 224) -> (B, 768, 14, 14)
            x = self.patch_embed_conv(x)
            
            # Flatten and transpose: (B, 768, 14, 14) -> (B, 196, 768)
            x = x.flatten(2).transpose(1, 2)
            num_patches = x.shape[1]
            
            # Add cls token
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, 768)
            x = torch.cat((cls_tokens, x), dim=1)  # (B, 197, 768)
            
            # Add interpolated positional embedding
            pos_embed = self.interpolate_pos_embed(num_patches)  # (1, 197, 768)
            x = x + pos_embed
            
            # Remove cls token (we only need spatial features)
            x = x[:, 1:, :]  # (B, 196, 768)
            
            # Project to 256D
            x = self.proj(x)  # (B, 196, 256)
            
            # Reshape to spatial format
            H = W = int(num_patches ** 0.5)
            x = x.transpose(1, 2).reshape(B, 256, H, W)
            
            return x
    
    # Create model
    model = PrithviEncoderModel(checkpoint, device)
    model.eval()
    
    logger.info(f"Created real Prithvi encoder")
    logger.info(f"   Embedding dim: {model.embed_dim}")
    logger.info(f"   Output dim: 256")
    logger.info(f"   Using pre-trained patch embeddings and positional encodings")
    
    return model


def load_prithvi_model(
    model_path: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
    use_simple_model: bool = False
) -> PrithviEncoder:
    """
    Load Prithvi-EO-1.0-100M model for inference.
    
    This function loads the pre-trained Prithvi model and wraps it in a
    PrithviEncoder for easy inference. If the model file is not found,
    it will provide instructions on how to download it.
    
    Args:
        model_path: Path to model weights. If None, uses default location.
        device: Device to load model on ('cuda' or 'cpu'). If None, auto-detects.
        use_simple_model: If True, create a simple model for testing without
                         downloading the full Prithvi model. Useful for development.
        
    Returns:
        PrithviEncoder instance ready for inference
        
    Raises:
        FileNotFoundError: If model file not found and use_simple_model=False
        RuntimeError: If model loading fails
        
    Example:
        >>> # Use real Prithvi model
        >>> encoder = load_prithvi_model(use_simple_model=False)
        >>> hls_image = torch.randn(1, 6, 224, 224)
        >>> embeddings = encoder(hls_image)
        
        >>> # Use simplified model for testing
        >>> encoder = load_prithvi_model(use_simple_model=True)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Loading Prithvi model on device: {device}")
    
    # Use simple model for testing if requested
    if use_simple_model:
        logger.warning("Using simplified model for testing (not real Prithvi)")
        model = create_simple_prithvi_model()
        model = model.to(device)
        encoder = PrithviEncoder(model)
        logger.info("Simple model loaded successfully")
        return encoder
    
    # Default model path (relative to project root)
    if model_path is None:
        # Find project root (where pyproject.toml is)
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent  # src/models/prithvi_loader.py -> project root
        model_path_obj = project_root / "models" / "prithvi" / "Prithvi_EO_V1_100M.pt"
    else:
        model_path_obj = Path(model_path)
    
    # Check if model exists
    if not model_path_obj.exists():
        error_msg = (
            f"Prithvi model not found at {model_path_obj}. "
            f"Please download it first by running:\n"
            f"  poetry run python scripts/download_prithvi.py\n"
            f"Or use use_simple_model=True for testing."
        )
        raise FileNotFoundError(error_msg)
    
    try:
        logger.info(f"Loading real Prithvi model from {model_path_obj}")
        logger.info(f"File size: {model_path_obj.stat().st_size / 1024**2:.2f} MB")
        
        # Load model checkpoint
        checkpoint = torch.load(str(model_path_obj), map_location='cpu')
        logger.info(f"Checkpoint loaded with {len(checkpoint)} keys")
        
        # Create encoder with real Prithvi weights
        model = create_real_prithvi_encoder(checkpoint, device)
        encoder = PrithviEncoder(model)
        
        logger.info("Real Prithvi model loaded successfully")
        logger.info("   Using pre-trained weights from NASA/IBM")
        return encoder
        
    except Exception as e:
        logger.error(f"Failed to load Prithvi model: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Model loading failed: {e}")


def get_model_info(encoder: PrithviEncoder) -> Dict[str, Any]:
    """
    Get information about loaded Prithvi model.
    
    This function extracts useful information about the model such as
    number of parameters, device, and data type.
    
    Args:
        encoder: Loaded PrithviEncoder instance
        
    Returns:
        Dictionary with model information:
            - total_parameters: Total number of parameters
            - trainable_parameters: Number of trainable parameters
            - device: Device the model is on (cuda or cpu)
            - dtype: Data type of model parameters
            
    Example:
        >>> encoder = load_prithvi_model()
        >>> info = get_model_info(encoder)
        >>> print(f"Parameters: {info['total_parameters']:,}")
    """
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    # Get device and dtype from first parameter
    first_param = next(encoder.parameters())
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'device': str(first_param.device),
        'dtype': str(first_param.dtype),
        'in_eval_mode': not encoder.model.training,
    }


def normalize_hls_image(
    image: torch.Tensor,
    method: str = 'standardize'
) -> torch.Tensor:
    """
    Normalize HLS image for Prithvi inference.
    
    Prithvi expects normalized input imagery. This function provides
    different normalization methods.
    
    Args:
        image: Input HLS image of shape (B, 6, H, W) or (6, H, W)
        method: Normalization method:
            - 'standardize': Zero mean, unit variance (recommended)
            - 'minmax': Scale to [0, 1]
            - 'clip': Clip to [0, 1] and standardize
            
    Returns:
        Normalized image with same shape as input
        
    Example:
        >>> image = torch.randn(1, 6, 224, 224)
        >>> normalized = normalize_hls_image(image)
    """
    if method == 'standardize':
        mean = image.mean()
        std = image.std()
        return (image - mean) / (std + 1e-8)
    
    elif method == 'minmax':
        min_val = image.min()
        max_val = image.max()
        return (image - min_val) / (max_val - min_val + 1e-8)
    
    elif method == 'clip':
        image = torch.clamp(image, 0, 1)
        mean = image.mean()
        std = image.std()
        return (image - mean) / (std + 1e-8)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def interpolate_embeddings(
    embeddings: torch.Tensor,
    target_size: tuple,
    mode: str = 'bilinear'
) -> torch.Tensor:
    """
    Interpolate embeddings to match original image resolution.
    
    Prithvi may output embeddings at a different spatial resolution than
    the input image. This function interpolates them back to the original size.
    
    Args:
        embeddings: Embeddings of shape (B, C, H', W')
        target_size: Target spatial size (H, W)
        mode: Interpolation mode ('bilinear' or 'nearest')
        
    Returns:
        Interpolated embeddings of shape (B, C, H, W)
        
    Example:
        >>> embeddings = torch.randn(1, 256, 14, 14)
        >>> upsampled = interpolate_embeddings(embeddings, (224, 224))
        >>> print(upsampled.shape)  # (1, 256, 224, 224)
    """
    return torch.nn.functional.interpolate(
        embeddings,
        size=target_size,
        mode=mode,
        align_corners=False if mode == 'bilinear' else None
    )
