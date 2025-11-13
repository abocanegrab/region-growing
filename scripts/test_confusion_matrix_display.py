"""
Test script to verify confusion matrix display works correctly.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append('.')

from src.utils.validation_metrics import (
    generate_confusion_matrix,
    plot_confusion_matrix
)

# Create dummy data
np.random.seed(42)
prediction = np.random.randint(0, 5, size=(100, 100))
ground_truth = np.random.randint(0, 5, size=(100, 100))

# Generate confusion matrix
cm = generate_confusion_matrix(prediction, ground_truth, num_classes=5)

# Class names
class_names = ['Water', 'Crop', 'Urban', 'Bare Soil', 'Other']

# Test 1: Save only
print("Test 1: Saving confusion matrix...")
fig = plot_confusion_matrix(
    cm,
    class_names,
    save_path='img/results/validation/test_cm.png',
    normalize=False,
    title="Test Confusion Matrix"
)
print("✅ Saved successfully")
plt.close(fig)

# Test 2: Display in interactive mode
print("\nTest 2: Displaying confusion matrix...")
fig = plot_confusion_matrix(
    cm,
    class_names,
    save_path=None,
    normalize=False,
    title="Test Confusion Matrix (Display)"
)
plt.show()
print("✅ If you see the figure above, display works!")
plt.close(fig)

print("\n✅ All tests completed")
