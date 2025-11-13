"""
Analyze Dynamic World data to understand class distribution.
"""
import sys
from pathlib import Path
import numpy as np
import rasterio

def main():
    print("=" * 70)
    print("ANÁLISIS DE DYNAMIC WORLD")
    print("=" * 70)
    
    # Dynamic World class names (English/Spanish)
    DW_CLASSES = {
        0: "Water (Agua)",
        1: "Trees (Árboles)",
        2: "Grass (Pasto)",
        3: "Flooded Vegetation (Vegetación Inundada)",
        4: "Crops (Cultivos)",
        5: "Shrub & Scrub (Arbustos y Matorrales)",
        6: "Built Area (Área Construida)",
        7: "Bare Ground (Suelo Desnudo)",
        8: "Snow & Ice (Nieve y Hielo)"
    }
    
    zones = ["mexicali", "bajio", "sinaloa"]
    
    for zone in zones:
        print(f"\n--- {zone.upper()} ---")
        dw_path = Path(f"data/dynamic_world/{zone}_dw.tif")
        
        if not dw_path.exists():
            print(f"  ✗ No encontrado: {dw_path}")
            continue
        
        with rasterio.open(dw_path) as src:
            dw_mask = src.read(1)
            print(f"  Shape: {dw_mask.shape}")
            print(f"  Dtype: {dw_mask.dtype}")
            
            # Count pixels per class
            unique, counts = np.unique(dw_mask, return_counts=True)
            total_pixels = dw_mask.size
            
            print(f"\n  Distribución de clases:")
            print(f"  {'Clase':<25} {'Píxeles':>12} {'%':>8}")
            print("  " + "-" * 50)
            
            for class_id, count in zip(unique, counts):
                class_name = DW_CLASSES.get(class_id, f"Unknown ({class_id})")
                percentage = (count / total_pixels) * 100
                print(f"  {class_name:<25} {count:>12,} {percentage:>7.2f}%")
            
            print("  " + "-" * 50)
            print(f"  {'TOTAL':<25} {total_pixels:>12,} {100.0:>7.2f}%")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
