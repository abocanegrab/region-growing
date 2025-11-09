"""
Test both simple and real Prithvi models.
"""
import torch
import time
import logging

from src.models.prithvi_loader import load_prithvi_model, get_model_info

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_model(use_simple: bool):
    """Test a model configuration."""
    model_type = "Simple" if use_simple else "Real"
    logger.info("\n" + "="*70)
    logger.info(f"Testing {model_type} Prithvi Model")
    logger.info("="*70)
    
    # Load model
    encoder = load_prithvi_model(use_simple_model=use_simple)
    
    # Get info
    info = get_model_info(encoder)
    logger.info(f"\nModel Info:")
    logger.info(f"  Parameters: {info['total_parameters']:,}")
    logger.info(f"  Device: {info['device']}")
    
    # Test inference
    test_input = torch.randn(2, 6, 224, 224)  # Batch of 2
    device = next(encoder.parameters()).device
    test_input = test_input.to(device)
    
    start = time.time()
    with torch.no_grad():
        embeddings = encoder(test_input)
    elapsed = time.time() - start
    
    logger.info(f"\nInference:")
    logger.info(f"  Input: {tuple(test_input.shape)}")
    logger.info(f"  Output: {tuple(embeddings.shape)}")
    logger.info(f"  Time: {elapsed:.3f}s ({elapsed/2:.3f}s per image)")
    logger.info(f"  Range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
    
    # Verify output shape
    assert embeddings.shape[0] == 2, "Batch size should be 2"
    assert embeddings.shape[1] == 256, "Should have 256 embedding channels"
    assert embeddings.ndim == 4, "Should be 4D tensor (B, C, H, W)"
    
    logger.info(f"\n{model_type} model works correctly!")
    
    return embeddings

def main():
    logger.info("\n" + "#"*70)
    logger.info("# Testing Both Model Configurations")
    logger.info("#"*70)
    
    # Test simple model
    embeddings_simple = test_model(use_simple=True)
    
    # Test real model
    embeddings_real = test_model(use_simple=False)
    
    # Compare outputs
    logger.info("\n" + "="*70)
    logger.info("Comparison")
    logger.info("="*70)
    logger.info(f"Simple model output range: [{embeddings_simple.min():.3f}, {embeddings_simple.max():.3f}]")
    logger.info(f"Real model output range: [{embeddings_real.min():.3f}, {embeddings_real.max():.3f}]")
    logger.info(f"Simple model std: {embeddings_simple.std():.3f}")
    logger.info(f"Real model std: {embeddings_real.std():.3f}")
    
    logger.info("\n" + "="*70)
    logger.info("ALL TESTS PASSED")
    logger.info("="*70)
    logger.info("\nBoth models are working correctly!")
    logger.info("- use_simple_model=True: Fast, for development")
    logger.info("- use_simple_model=False: Real Prithvi, for production")

if __name__ == "__main__":
    main()
