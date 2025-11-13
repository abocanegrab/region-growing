#!/usr/bin/env python3
"""
Command-line interface for hierarchical analysis.

This script provides a simple CLI for executing the complete analysis pipeline
from Sentinel-2 download to classified stress maps.

Usage:
    python analyze_region.py --bbox "32.45,-115.35,32.55,-115.25" --date "2025-10-15"
    python analyze_region.py --bbox "32.45,-115.35,32.55,-115.25" --date "2025-10-15" --output "output/mexicali"
    python analyze_region.py --help

Examples:
    # Basic analysis
    python analyze_region.py --bbox "32.45,-115.35,32.55,-115.25" --date "2025-10-15"

    # Custom threshold
    python analyze_region.py --bbox "..." --date "..." --threshold 0.90

    # Export specific formats
    python analyze_region.py --bbox "..." --date "..." --formats json,tif

    # Verbose logging
    python analyze_region.py --bbox "..." --date "..." --verbose
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.hierarchical_analysis import (
    HierarchicalAnalysisPipeline,
    AnalysisConfig
)


def parse_bbox(bbox_str: str) -> Tuple[float, float, float, float]:
    """
    Parse bbox string to tuple.

    Parameters
    ----------
    bbox_str : str
        Bbox as "min_lat,min_lon,max_lat,max_lon"

    Returns
    -------
    tuple
        (min_lon, min_lat, max_lon, max_lat)

    Raises
    ------
    ValueError
        If bbox format is invalid

    Examples
    --------
    >>> bbox = parse_bbox("32.45,-115.35,32.55,-115.25")
    >>> print(bbox)
    (-115.35, 32.45, -115.25, 32.55)
    """
    try:
        coords = [float(x.strip()) for x in bbox_str.split(',')]
        if len(coords) != 4:
            raise ValueError("BBox must have 4 coordinates")

        # Convert from lat,lon to lon,lat format
        min_lat, min_lon, max_lat, max_lon = coords
        return (min_lon, min_lat, max_lon, max_lat)
    except Exception as e:
        raise ValueError(f"Invalid bbox format: {e}")


def setup_logging(verbose: bool = False):
    """
    Setup logging configuration.

    Parameters
    ----------
    verbose : bool
        Enable verbose (DEBUG) logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('analysis.log')
        ]
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Hierarchical Analysis CLI - Region Growing System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze Mexicali region
  python analyze_region.py --bbox "32.45,-115.35,32.55,-115.25" --date "2025-10-15"

  # Custom output directory
  python analyze_region.py --bbox "..." --date "..." --output "results/mexicali_20251015"

  # Export only JSON and PNG
  python analyze_region.py --bbox "..." --date "..." --formats json,png

  # Adjust MGRG threshold
  python analyze_region.py --bbox "..." --date "..." --threshold 0.90

For more information: https://github.com/abocanegrab/region-growing
        """
    )

    # Required arguments
    parser.add_argument(
        '--bbox',
        type=str,
        required=True,
        help='Bounding box as "min_lat,min_lon,max_lat,max_lon" (WGS84)'
    )
    parser.add_argument(
        '--date',
        type=str,
        required=True,
        help='Analysis date as YYYY-MM-DD (e.g., 2025-10-15)'
    )

    # Optional arguments
    parser.add_argument(
        '--output',
        type=str,
        default='output/cli',
        help='Output directory (default: output/cli)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.95,
        help='MGRG similarity threshold 0.7-0.99 (default: 0.95)'
    )
    parser.add_argument(
        '--min-size',
        type=int,
        default=50,
        help='Minimum region size in pixels (default: 50)'
    )
    parser.add_argument(
        '--formats',
        type=str,
        default='json,tif,png',
        help='Export formats comma-separated: json,tif,png,html (default: json,tif,png)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Parse inputs
        bbox = parse_bbox(args.bbox)
        export_formats = [f.strip() for f in args.formats.split(',')]

        # Create config
        config = AnalysisConfig(
            bbox=bbox,
            date_from=args.date,
            mgrg_threshold=args.threshold,
            min_region_size=args.min_size,
            output_dir=args.output,
            export_formats=export_formats
        )

        logger.info("="*60)
        logger.info("Hierarchical Analysis CLI")
        logger.info("="*60)
        logger.info(f"BBox: {bbox}")
        logger.info(f"Date: {args.date}")
        logger.info(f"Output: {args.output}")
        logger.info(f"Threshold: {args.threshold}")
        logger.info(f"Formats: {export_formats}")
        logger.info("="*60)

        # Run pipeline
        pipeline = HierarchicalAnalysisPipeline(config)
        result = pipeline.run()

        # Print results
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"\nOutput files:")
        for file_type, file_path in result.output_files.items():
            print(f"  {file_type.upper()}: {file_path}")

        print(f"\nSummary:")
        for key, value in result.summary.items():
            print(f"  {key}: {value:.1f} ha")

        print(f"\nProcessing time:")
        for step, time_s in result.processing_time.items():
            print(f"  {step}: {time_s}s")

        print("\n" + "="*60)

        return 0  # Success

    except KeyboardInterrupt:
        logger.warning("Analysis interrupted by user")
        return 130  # SIGINT

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        print(f"\nERROR: {e}", file=sys.stderr)
        print("Use --verbose for detailed error information", file=sys.stderr)
        return 1  # Error


if __name__ == "__main__":
    sys.exit(main())
