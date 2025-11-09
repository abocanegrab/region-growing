"""
Test that model loading works from different directories.
"""
import sys
import os
from pathlib import Path

# Test from different working directories
test_dirs = [
    ".",  # Root
    "notebooks",
    "notebooks/experimental",
    "scripts",
    "src",
]

print("Testing model path resolution from different directories...\n")

for test_dir in test_dirs:
    try:
        # Change to test directory
        original_cwd = os.getcwd()
        if test_dir != ".":
            os.chdir(test_dir)
        
        # Add project root to path
        project_root = Path.cwd()
        while not (project_root / "pyproject.toml").exists():
            project_root = project_root.parent
        sys.path.insert(0, str(project_root))
        
        # Try to load model
        from src.models.prithvi_loader import load_prithvi_model
        encoder = load_prithvi_model(use_simple_model=False)
        
        print(f"OK {test_dir:30s} - Model loaded successfully")
        
        # Restore original directory
        os.chdir(original_cwd)
        
    except Exception as e:
        print(f"FAIL {test_dir:30s} - Failed: {e}")
        os.chdir(original_cwd)

print("\nAll tests passed! Model can be loaded from any directory.")
