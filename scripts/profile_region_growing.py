"""
Profiling script for Region Growing algorithm performance analysis.

This script benchmarks the Classic Region Growing algorithm across different
image sizes to measure performance characteristics.
"""
import time
import numpy as np
from src.algorithms.classic_region_growing import ClassicRegionGrowing


def generate_synthetic_ndvi(height: int, width: int, complexity: str = 'medium') -> np.ndarray:
    """
    Generate synthetic NDVI image with varying complexity.

    Args:
        height: Image height in pixels
        width: Image width in pixels
        complexity: Image complexity ('simple', 'medium', 'complex')

    Returns:
        2D NumPy array with synthetic NDVI values
    """
    if complexity == 'simple':
        # Mostly homogeneous with few variations
        image = np.random.rand(height, width) * 0.2 + 0.5
    elif complexity == 'medium':
        # Moderate variation typical of agricultural scenes
        image = np.random.rand(height, width) * 0.6 + 0.2
    elif complexity == 'complex':
        # High variation with many small regions
        image = np.random.rand(height, width) * 0.8 + 0.1
    else:
        raise ValueError(f"Invalid complexity: {complexity}")

    return image


def profile_algorithm(
    sizes: list = None,
    threshold: float = 0.1,
    min_region_size: int = 50
) -> None:
    """
    Profile Region Growing algorithm with different image sizes.

    Args:
        sizes: List of image sizes to test (default: [128, 256, 512, 1024])
        threshold: Homogeneity threshold for algorithm
        min_region_size: Minimum region size in pixels
    """
    if sizes is None:
        sizes = [128, 256, 512, 1024]

    print("=" * 70)
    print("Region Growing Algorithm Performance Profiling")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  - Threshold: {threshold}")
    print(f"  - Min region size: {min_region_size} pixels")
    print(f"\nImage complexity: Medium (typical agricultural scene)")
    print()

    results = []

    for size in sizes:
        # Generate synthetic NDVI image
        print(f"Testing {size}x{size} image...")
        ndvi = generate_synthetic_ndvi(size, size, complexity='medium')

        # Initialize algorithm
        algorithm = ClassicRegionGrowing(
            threshold=threshold,
            min_region_size=min_region_size
        )

        # Warm-up run (ignore results)
        algorithm.segment(ndvi)

        # Timed run
        start = time.time()
        labeled, num_regions, regions_info = algorithm.segment(ndvi)
        elapsed = time.time() - start

        # Calculate metrics
        total_pixels = size * size
        pixels_per_second = total_pixels / elapsed
        ms_per_megapixel = (elapsed * 1000) / (total_pixels / 1_000_000)

        # Store results
        result = {
            'size': size,
            'total_pixels': total_pixels,
            'time_seconds': elapsed,
            'num_regions': num_regions,
            'pixels_per_second': pixels_per_second,
            'ms_per_megapixel': ms_per_megapixel
        }
        results.append(result)

        # Print results
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Regions found: {num_regions}")
        print(f"  Throughput: {pixels_per_second:,.0f} pixels/sec")
        print(f"  Performance: {ms_per_megapixel:.2f} ms/Mpixel")
        print()

    # Summary table
    print("=" * 70)
    print("Performance Summary")
    print("=" * 70)
    print(f"{'Size':<12} {'Pixels':<12} {'Time':<10} {'Regions':<10} {'Pixels/sec':<15} {'ms/Mpixel':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['size']}x{r['size']:<6} {r['total_pixels']:<12,} {r['time_seconds']:<10.3f} "
              f"{r['num_regions']:<10} {r['pixels_per_second']:<15,.0f} {r['ms_per_megapixel']:<12.2f}")

    # Average throughput
    avg_throughput = np.mean([r['pixels_per_second'] for r in results])
    print("-" * 70)
    print(f"Average throughput: {avg_throughput:,.0f} pixels/sec")
    print()


def profile_different_thresholds() -> None:
    """Profile algorithm with different threshold values."""
    print("\n" + "=" * 70)
    print("Threshold Sensitivity Analysis (512x512 image)")
    print("=" * 70)
    print()

    size = 512
    ndvi = generate_synthetic_ndvi(size, size, complexity='medium')
    thresholds = [0.05, 0.08, 0.10, 0.12, 0.15]

    print(f"{'Threshold':<12} {'Time':<10} {'Regions':<10} {'Avg Size':<12}")
    print("-" * 50)

    for threshold in thresholds:
        algorithm = ClassicRegionGrowing(threshold=threshold, min_region_size=50)

        start = time.time()
        labeled, num_regions, regions_info = algorithm.segment(ndvi)
        elapsed = time.time() - start

        avg_region_size = np.mean([r['size'] for r in regions_info]) if num_regions > 0 else 0

        print(f"{threshold:<12.2f} {elapsed:<10.3f} {num_regions:<10} {avg_region_size:<12.0f}")

    print()


def profile_complexity_levels() -> None:
    """Profile algorithm with different image complexities."""
    print("\n" + "=" * 70)
    print("Image Complexity Analysis (512x512 image)")
    print("=" * 70)
    print()

    size = 512
    complexities = ['simple', 'medium', 'complex']

    print(f"{'Complexity':<12} {'Time':<10} {'Regions':<10} {'Regions/sec':<15}")
    print("-" * 55)

    for complexity in complexities:
        ndvi = generate_synthetic_ndvi(size, size, complexity=complexity)
        algorithm = ClassicRegionGrowing(threshold=0.1, min_region_size=50)

        start = time.time()
        labeled, num_regions, regions_info = algorithm.segment(ndvi)
        elapsed = time.time() - start

        regions_per_second = num_regions / elapsed

        print(f"{complexity:<12} {elapsed:<10.3f} {num_regions:<10} {regions_per_second:<15.0f}")

    print()


def main():
    """Run all profiling benchmarks."""
    print("\n" + "#" * 70)
    print("# Classic Region Growing Algorithm - Performance Profiling")
    print("#" * 70)
    print()

    # Main profiling across different sizes
    profile_algorithm()

    # Threshold sensitivity
    profile_different_thresholds()

    # Complexity analysis
    profile_complexity_levels()

    print("=" * 70)
    print("Profiling Complete!")
    print("=" * 70)
    print("\nNote: Performance varies based on:")
    print("  - CPU/GPU hardware specifications")
    print("  - Image content and complexity")
    print("  - Threshold and min_region_size parameters")
    print("  - System load and background processes")
    print()


if __name__ == "__main__":
    main()
