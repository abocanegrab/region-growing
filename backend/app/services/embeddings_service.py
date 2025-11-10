"""
Embeddings Service for HLS image processing.

This service acts as a thin wrapper around the hls_processor module,
providing a clean interface for the FastAPI endpoints.
"""
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import uuid
from datetime import datetime

from src.utils.sentinel_download import create_sentinel_config, download_hls_bands
from src.features.hls_processor import (
    prepare_hls_image,
    extract_embeddings,
    save_embeddings,
    load_embeddings,
    compute_cosine_similarity
)

logger = logging.getLogger(__name__)


class EmbeddingsService:
    """
    Service for managing HLS embeddings extraction and storage.
    
    This service handles:
    - Downloading HLS bands from Sentinel Hub
    - Extracting embeddings using Prithvi model
    - Saving/loading embeddings from disk
    - Computing similarity between embeddings
    """
    
    def __init__(
        self,
        sentinel_client_id: str,
        sentinel_client_secret: str,
        embeddings_dir: Optional[Path] = None,
        use_simple_model: bool = False
    ):
        """
        Initialize embeddings service.
        
        Parameters
        ----------
        sentinel_client_id : str
            Sentinel Hub client ID
        sentinel_client_secret : str
            Sentinel Hub client secret
        embeddings_dir : Path, optional
            Directory to store embeddings. If None, uses default.
        use_simple_model : bool, default=False
            If True, uses simplified model for testing
        """
        self.sentinel_config = create_sentinel_config(
            sentinel_client_id,
            sentinel_client_secret
        )
        
        if embeddings_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent
            embeddings_dir = project_root / 'img' / 'sentinel2' / 'embeddings'
        
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_simple_model = use_simple_model
        
        logger.info(f"EmbeddingsService initialized")
        logger.info(f"  Embeddings directory: {self.embeddings_dir}")
        logger.info(f"  Using {'simple' if use_simple_model else 'Prithvi'} model")
    
    def extract_embeddings_from_bbox(
        self,
        bbox_coords: Dict[str, float],
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        max_cloud_coverage: float = 0.3
    ) -> Tuple[np.ndarray, Dict]:
        """
        Extract embeddings from a geographic bounding box.
        
        This method:
        1. Downloads HLS bands from Sentinel Hub
        2. Prepares HLS image
        3. Extracts embeddings using Prithvi
        
        Parameters
        ----------
        bbox_coords : dict
            Bounding box with keys: min_lat, min_lon, max_lat, max_lon
        date_from : str, optional
            Start date in YYYY-MM-DD format
        date_to : str, optional
            End date in YYYY-MM-DD format
        max_cloud_coverage : float, default=0.3
            Maximum cloud coverage (0.0 to 1.0)
            
        Returns
        -------
        embeddings : np.ndarray
            Extracted embeddings with shape (H, W, 256)
        metadata : dict
            Metadata about the extraction
        """
        logger.info(f"Extracting embeddings for bbox: {bbox_coords}")
        
        logger.info("Downloading HLS bands from Sentinel Hub...")
        data = download_hls_bands(
            bbox_coords=bbox_coords,
            config=self.sentinel_config,
            date_from=date_from,
            date_to=date_to,
            max_cloud_coverage=max_cloud_coverage
        )
        
        logger.info("Preparing HLS image...")
        hls_image = prepare_hls_image(
            data['bands_10m'],
            data['bands_20m']
        )
        
        logger.info("Extracting embeddings...")
        embeddings = extract_embeddings(
            hls_image,
            use_simple_model=self.use_simple_model,
            normalize_output=True
        )
        
        metadata = {
            'bbox': bbox_coords,
            'date_from': data['metadata']['date_from'],
            'date_to': data['metadata']['date_to'],
            'dimensions': data['metadata']['dimensions_10m'],
            'embeddings_shape': embeddings.shape,
            'model': 'simple' if self.use_simple_model else 'prithvi',
            'extracted_at': datetime.now().isoformat()
        }
        
        logger.info(f"Embeddings extracted successfully: {embeddings.shape}")
        return embeddings, metadata
    
    def save_embeddings_to_disk(
        self,
        embeddings: np.ndarray,
        metadata: Dict
    ) -> Tuple[str, Path]:
        """
        Save embeddings to disk with unique ID.
        
        Parameters
        ----------
        embeddings : np.ndarray
            Embeddings to save
        metadata : dict
            Metadata to save alongside embeddings
            
        Returns
        -------
        embedding_id : str
            Unique ID for the saved embeddings
        file_path : Path
            Path to saved file
        """
        embedding_id = str(uuid.uuid4())
        file_path = self.embeddings_dir / f"{embedding_id}.npz"
        
        save_embeddings(embeddings, file_path, metadata)
        
        logger.info(f"Saved embeddings with ID: {embedding_id}")
        return embedding_id, file_path
    
    def load_embeddings_from_disk(
        self,
        embedding_id: str
    ) -> Tuple[np.ndarray, Dict]:
        """
        Load embeddings from disk by ID.
        
        Parameters
        ----------
        embedding_id : str
            Unique ID of embeddings to load
            
        Returns
        -------
        embeddings : np.ndarray
            Loaded embeddings
        metadata : dict
            Loaded metadata
        """
        file_path = self.embeddings_dir / f"{embedding_id}.npz"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Embeddings not found: {embedding_id}")
        
        return load_embeddings(file_path)
    
    def compute_similarity_between_regions(
        self,
        bbox_a: Dict[str, float],
        bbox_b: Dict[str, float],
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> Dict:
        """
        Compute similarity between two geographic regions.
        
        Parameters
        ----------
        bbox_a : dict
            First bounding box
        bbox_b : dict
            Second bounding box
        date_from : str, optional
            Start date
        date_to : str, optional
            End date
            
        Returns
        -------
        dict
            Dictionary with similarity metrics and statistics
        """
        logger.info("Computing similarity between two regions")
        
        embeddings_a, metadata_a = self.extract_embeddings_from_bbox(
            bbox_a, date_from, date_to
        )
        
        embeddings_b, metadata_b = self.extract_embeddings_from_bbox(
            bbox_b, date_from, date_to
        )
        
        min_h = min(embeddings_a.shape[0], embeddings_b.shape[0])
        min_w = min(embeddings_a.shape[1], embeddings_b.shape[1])
        
        embeddings_a_crop = embeddings_a[:min_h, :min_w, :]
        embeddings_b_crop = embeddings_b[:min_h, :min_w, :]
        
        similarity_map = compute_cosine_similarity(
            embeddings_a_crop,
            embeddings_b_crop
        )
        
        return {
            'mean_similarity': float(similarity_map.mean()),
            'std_similarity': float(similarity_map.std()),
            'min_similarity': float(similarity_map.min()),
            'max_similarity': float(similarity_map.max()),
            'similarity_map_shape': similarity_map.shape,
            'region_a_metadata': metadata_a,
            'region_b_metadata': metadata_b
        }
    
    def get_embeddings_file_path(self, embedding_id: str) -> Path:
        """
        Get file path for embeddings by ID.
        
        Parameters
        ----------
        embedding_id : str
            Unique ID of embeddings
            
        Returns
        -------
        Path
            File path to embeddings
        """
        return self.embeddings_dir / f"{embedding_id}.npz"
