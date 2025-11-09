"""
Script to download Prithvi-EO-1.0-100M model from HuggingFace.

This script downloads the Prithvi foundation model pre-trained by NASA and IBM
for geospatial AI applications. The model is used to extract semantic embeddings
from Sentinel-2 imagery.

Usage:
    poetry run python scripts/download_prithvi.py
    
References:
    Jakubik et al. (2024). Foundation models for generalist geospatial AI.
    https://arxiv.org/abs/2310.18660
"""
import logging
from pathlib import Path
from typing import Optional
from huggingface_hub import hf_hub_download  # type: ignore

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def download_prithvi_model(
    repo_id: str = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M",
    model_dir: str = "models/prithvi",
    filename: str = "Prithvi_EO_V1_100M.pt"
) -> Path:
    """
    Download Prithvi model from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repository ID
        model_dir: Local directory to save model
        filename: Name of the model file
        
    Returns:
        Path to downloaded model file
        
    Raises:
        Exception: If download fails
    """
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading Prithvi model from {repo_id}...")
    logger.info(f"Target directory: {model_path.absolute()}")
    logger.info("This may take several minutes (~400 MB)...")
    
    try:
        model_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(model_path),
            local_dir=str(model_path),
            local_dir_use_symlinks=False
        )
        
        logger.info(f"Model downloaded successfully to: {model_file}")
        
        # Verify file size
        file_size = Path(model_file).stat().st_size / (1024 * 1024)  # MB
        logger.info(f"Model file size: {file_size:.2f} MB")
        
        return Path(model_file)
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise


def download_model_config(
    repo_id: str = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M",
    model_dir: str = "models/prithvi",
    filename: str = "config.yaml"
) -> Optional[Path]:
    """
    Download Prithvi model configuration file.
    
    Args:
        repo_id: HuggingFace repository ID
        model_dir: Local directory to save config
        filename: Name of the config file
        
    Returns:
        Path to downloaded config file
    """
    model_path = Path(model_dir)
    
    try:
        logger.info("Downloading config file...")
        
        config_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(model_path),
            local_dir=str(model_path),
            local_dir_use_symlinks=False
        )
        
        logger.info(f"Config downloaded to: {config_file}")
        return Path(config_file)
        
    except Exception as e:
        logger.warning(f"Config file not found or failed to download: {e}")
        logger.warning("Continuing without config file (will use defaults)")
        return None


def main():
    """Main function to download Prithvi model and config."""
    logger.info("="*70)
    logger.info("Prithvi Model Download Script")
    logger.info("="*70)
    
    try:
        # Download model
        model_path = download_prithvi_model()
        
        # Download config (optional)
        config_path = download_model_config()
        
        logger.info("="*70)
        logger.info("Download Complete!")
        logger.info("="*70)
        logger.info(f"Model: {model_path}")
        if config_path:
            logger.info(f"Config: {config_path}")
        logger.info("\nYou can now use the model in your code:")
        logger.info("  from src.models.prithvi_loader import load_prithvi_model")
        logger.info("  encoder = load_prithvi_model()")
        
    except Exception as e:
        logger.error("Download Failed!")
        logger.error(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
