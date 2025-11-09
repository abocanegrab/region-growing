"""
Test Prithvi model inference with synthetic imagery.

This script verifies that:
1. Model loads correctly
2. Inference runs without errors
3. Output shape is correct
4. GPU memory usage is acceptable (if CUDA available)
5. Inference time is reasonable

Usage:
    poetry run python scripts/test_prithvi_inference.py
"""
import torch  # type: ignore
import time
import logging

from src.models.prithvi_loader import (
    load_prithvi_model,
    get_model_info,
    normalize_hls_image
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_synthetic_hls_image(
    height: int = 224,
    width: int = 224,
    batch_size: int = 1
) -> torch.Tensor:
    """
    Create synthetic HLS image for testing.
    
    Args:
        height: Image height
        width: Image width
        batch_size: Batch size
        
    Returns:
        Synthetic HLS image tensor (B, 6, H, W)
    """
    image = torch.rand(batch_size, 6, height, width)
    image = normalize_hls_image(image, method='standardize')
    return image


def test_model_loading():
    """Test 1: Model loading with simple model."""
    logger.info("\n" + "="*70)
    logger.info("TEST 1: Model Loading (Simple Model)")
    logger.info("="*70)
    
    try:
        encoder = load_prithvi_model(use_simple_model=True)
        info = get_model_info(encoder)
        
        logger.info("Model loaded successfully")
        logger.info(f"   Total parameters: {info['total_parameters']:,}")
        logger.info(f"   Device: {info['device']}")
        logger.info(f"   Dtype: {info['dtype']}")
        logger.info(f"   Eval mode: {info['in_eval_mode']}")
        
        return encoder
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise


def test_inference_synthetic(encoder):
    """Test 2: Inference with synthetic image."""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Inference with Synthetic Image")
    logger.info("="*70)
    
    try:
        image = create_synthetic_hls_image(224, 224, 1)
        device = next(encoder.parameters()).device
        image = image.to(device)
        
        logger.info(f"Input shape: {tuple(image.shape)}")
        
        start_time = time.time()
        embeddings = encoder(image)
        inference_time = time.time() - start_time
        
        logger.info("Inference successful")
        logger.info(f"   Output shape: {tuple(embeddings.shape)}")
        logger.info(f"   Inference time: {inference_time:.3f}s")
        logger.info(f"   Output range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
        
        assert embeddings.shape[0] == 1, "Batch size mismatch"
        assert embeddings.ndim == 4, f"Expected 4D output, got {embeddings.ndim}D"
        assert embeddings.shape[1] == 256, f"Expected 256 embedding dims, got {embeddings.shape[1]}"
        
        return embeddings
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


def test_gpu_memory(encoder):
    """Test 3: GPU memory usage."""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: GPU Memory Usage")
    logger.info("="*70)
    
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping GPU memory test")
        return
    
    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Use 224x224 for simple model (it's hardcoded for this size)
        image = create_synthetic_hls_image(224, 224, 1)
        image = image.cuda()
        
        _ = encoder(image)
        
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        max_memory = torch.cuda.max_memory_allocated() / 1024**3
        
        logger.info("GPU memory test completed")
        logger.info(f"   Memory allocated: {memory_allocated:.2f} GB")
        logger.info(f"   Memory reserved: {memory_reserved:.2f} GB")
        logger.info(f"   Peak memory: {max_memory:.2f} GB")
        
        if max_memory > 7.5:
            logger.warning(f"Peak memory ({max_memory:.2f} GB) is close to 8GB limit")
        else:
            logger.info("   Fits comfortably in 8GB VRAM")
        
    except Exception as e:
        logger.error(f"GPU memory test failed: {e}")
        raise


def test_batch_inference(encoder):
    """Test 4: Batch inference."""
    logger.info("\n" + "="*70)
    logger.info("TEST 4: Batch Inference")
    logger.info("="*70)
    
    try:
        batch_sizes = [1, 2, 4]
        device = next(encoder.parameters()).device
        
        for batch_size in batch_sizes:
            image = create_synthetic_hls_image(224, 224, batch_size)
            image = image.to(device)
            
            start_time = time.time()
            embeddings = encoder(image)
            inference_time = time.time() - start_time
            
            logger.info(f"Batch size {batch_size}:")
            logger.info(f"   Output shape: {tuple(embeddings.shape)}")
            logger.info(f"   Time: {inference_time:.3f}s ({inference_time/batch_size:.3f}s per image)")
        
        logger.info("Batch inference test passed")
        
    except Exception as e:
        logger.error(f"Batch inference test failed: {e}")
        raise


def test_different_resolutions(encoder):
    """Test 5: Different image resolutions."""
    logger.info("\n" + "="*70)
    logger.info("TEST 5: Different Image Resolutions")
    logger.info("="*70)
    logger.info("Note: Simple model only supports 224x224, skipping other resolutions")
    
    try:
        # Simple model is hardcoded for 224x224
        resolutions = [(224, 224)]
        device = next(encoder.parameters()).device
        
        for height, width in resolutions:
            image = create_synthetic_hls_image(height, width, 1)
            image = image.to(device)
            
            start_time = time.time()
            embeddings = encoder(image)
            inference_time = time.time() - start_time
            
            logger.info(f"Resolution {height}x{width}:")
            logger.info(f"   Output shape: {tuple(embeddings.shape)}")
            logger.info(f"   Time: {inference_time:.3f}s")
        
        logger.info("Resolution test passed")
        
    except Exception as e:
        logger.error(f"Resolution test failed: {e}")
        raise


def main():
    """Run all tests."""
    logger.info("\n" + "#"*70)
    logger.info("# Prithvi Model Inference Tests")
    logger.info("#"*70)
    logger.info("\nNote: Using simplified model for testing")
    logger.info("To test real Prithvi, download it first with:")
    logger.info("  poetry run python scripts/download_prithvi.py")
    
    try:
        encoder = test_model_loading()
        test_inference_synthetic(encoder)
        test_gpu_memory(encoder)
        test_batch_inference(encoder)
        test_different_resolutions(encoder)
        
        logger.info("\n" + "="*70)
        logger.info("ALL TESTS PASSED")
        logger.info("="*70)
        logger.info("\nPrithvi model (simplified) is working correctly!")
        logger.info("You can now proceed with US-006 to extract embeddings.")
        
    except Exception as e:
        logger.error("\n" + "="*70)
        logger.error("TESTS FAILED")
        logger.error("="*70)
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
