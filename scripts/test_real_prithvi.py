"""
Test loading and using the real Prithvi model.
"""
import torch
import time
import logging

from src.models.prithvi_loader import load_prithvi_model, get_model_info

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("="*70)
    logger.info("Testing Real Prithvi Model")
    logger.info("="*70)
    
    # Load real model
    logger.info("\n1. Loading real Prithvi model...")
    encoder = load_prithvi_model(use_simple_model=False)
    
    # Get model info
    info = get_model_info(encoder)
    logger.info("\n2. Model Information:")
    logger.info(f"   Total parameters: {info['total_parameters']:,}")
    logger.info(f"   Device: {info['device']}")
    logger.info(f"   Dtype: {info['dtype']}")
    logger.info(f"   In eval mode: {info['in_eval_mode']}")
    
    # Test inference
    logger.info("\n3. Testing inference...")
    test_input = torch.randn(1, 6, 224, 224)
    device = next(encoder.parameters()).device
    test_input = test_input.to(device)
    
    start_time = time.time()
    with torch.no_grad():
        embeddings = encoder(test_input)
    inference_time = time.time() - start_time
    
    logger.info(f"   Input shape: {tuple(test_input.shape)}")
    logger.info(f"   Output shape: {tuple(embeddings.shape)}")
    logger.info(f"   Inference time: {inference_time:.3f}s")
    logger.info(f"   Output range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
    
    # Test GPU memory
    if torch.cuda.is_available():
        logger.info("\n4. GPU Memory Usage:")
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"   Memory allocated: {memory_allocated:.2f} GB")
        logger.info(f"   Memory reserved: {memory_reserved:.2f} GB")
    
    logger.info("\n" + "="*70)
    logger.info("Real Prithvi model is working!")
    logger.info("="*70)

if __name__ == "__main__":
    main()
