"""
Compare semantic similarity between Mexican agricultural zones.

This script loads embeddings from the 3 Mexican zones and calculates
cosine similarity between them.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.features.hls_processor import load_embeddings, compute_cosine_similarity


def main():
    print('=' * 70)
    print('ANÁLISIS DE SIMILITUD ENTRE ZONAS MEXICANAS')
    print('=' * 70)
    print()

    # Load embeddings
    zones = ['mexicali', 'bajio', 'sinaloa']
    embeddings_dict = {}

    print('Cargando embeddings...')
    for zone in zones:
        path = Path('img/sentinel2/embeddings') / f'{zone}_prithvi.npz'
        emb, meta = load_embeddings(path)
        embeddings_dict[zone] = emb
        print(f'  OK {zone.capitalize()}: {emb.shape}')

    print()
    print('Recortando al tamaño mínimo común...')

    # Crop all to same size for comparison
    min_h = min(emb.shape[0] for emb in embeddings_dict.values())
    min_w = min(emb.shape[1] for emb in embeddings_dict.values())

    print(f'  Tamaño común: ({min_h}, {min_w})')
    print()

    for zone in zones:
        embeddings_dict[zone] = embeddings_dict[zone][:min_h, :min_w, :]

    print('=' * 70)
    print('MATRIZ DE SIMILITUD COSENO')
    print('=' * 70)
    print()

    # Calculate similarities
    results = []
    for i, zone_a in enumerate(zones):
        for j, zone_b in enumerate(zones):
            if i < j:  # Only upper triangle
                print(f'Calculando similitud: {zone_a} vs {zone_b}...')
                sim_map = compute_cosine_similarity(
                    embeddings_dict[zone_a],
                    embeddings_dict[zone_b]
                )
                mean_sim = float(sim_map.mean())
                std_sim = float(sim_map.std())
                min_sim = float(sim_map.min())
                max_sim = float(sim_map.max())

                print(f'\n  [{zone_a.upper()} vs {zone_b.upper()}]:')
                print(f'    Mean:  {mean_sim:.4f}')
                print(f'    Std:   {std_sim:.4f}')
                print(f'    Range: [{min_sim:.4f}, {max_sim:.4f}]')

                # Interpretation
                if mean_sim > 0.9:
                    interp = 'MUY ALTA (mismo tipo de cobertura)'
                elif mean_sim > 0.7:
                    interp = 'ALTA (vegetacion similar)'
                elif mean_sim > 0.5:
                    interp = 'MEDIA (algunas similitudes)'
                else:
                    interp = 'BAJA (diferentes tipos de cobertura)'
                print(f'    ==> {interp}')
                print()

                results.append({
                    'zones': f'{zone_a}_vs_{zone_b}',
                    'mean': mean_sim,
                    'std': std_sim,
                    'min': min_sim,
                    'max': max_sim,
                    'interpretation': interp
                })

    print('=' * 70)
    print('RESUMEN')
    print('=' * 70)
    print()

    for result in results:
        print(f"{result['zones']}: {result['mean']:.4f} - {result['interpretation']}")

    print()
    print('=' * 70)
    print('OK - Analisis completado exitosamente')
    print('=' * 70)


if __name__ == '__main__':
    main()
