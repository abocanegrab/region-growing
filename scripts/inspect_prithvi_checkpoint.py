"""
Inspect Prithvi checkpoint structure.

This script loads the Prithvi checkpoint and prints its structure
to understand how to properly load it.
"""
import torch
from pathlib import Path

checkpoint_path = Path("models/prithvi/Prithvi_EO_V1_100M.pt")

print(f"Loading checkpoint from: {checkpoint_path}")
print(f"File size: {checkpoint_path.stat().st_size / 1024**2:.2f} MB\n")

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("Checkpoint keys:")
for key in checkpoint.keys():
    print(f"  - {key}")

print("\n" + "="*70)

# Check model structure
if 'model' in checkpoint:
    print("\nModel state_dict keys (first 20):")
    model_keys = list(checkpoint['model'].keys())
    for key in model_keys[:20]:
        print(f"  - {key}")
    print(f"\n  ... ({len(model_keys)} total keys)")
    
elif 'state_dict' in checkpoint:
    print("\nState dict keys (first 20):")
    state_keys = list(checkpoint['state_dict'].keys())
    for key in state_keys[:20]:
        print(f"  - {key}")
    print(f"\n  ... ({len(state_keys)} total keys)")

# Check for config
if 'config' in checkpoint:
    print("\nConfig:")
    print(checkpoint['config'])

# Check for metadata
if 'epoch' in checkpoint:
    print(f"\nEpoch: {checkpoint['epoch']}")
if 'arch' in checkpoint:
    print(f"Architecture: {checkpoint['arch']}")
