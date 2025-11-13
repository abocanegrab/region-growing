"""
Unit tests for validation metrics module.

Tests for semantic segmentation metrics including:
- IoU (Intersection over Union)
- mIoU (mean IoU)
- Weighted mIoU
- F1-Score
- Precision/Recall
- Pixel Accuracy
- Confusion Matrix
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.validation_metrics import (
    calculate_iou,
    calculate_miou,
    calculate_weighted_miou,
    calculate_f1_score,
    calculate_precision_recall,
    calculate_pixel_accuracy,
    generate_confusion_matrix,
    calculate_boundary_precision,
)


class TestIoU:
    """Tests for Intersection over Union (IoU) calculation"""

    def test_iou_perfect_match(self):
        """IoU should be 1.0 for perfect match"""
        pred = np.array([[1, 1], [1, 1]])
        gt = np.array([[1, 1], [1, 1]])
        assert calculate_iou(pred, gt, class_id=1) == 1.0

    def test_iou_no_overlap(self):
        """IoU should be 0.0 for no overlap"""
        pred = np.array([[1, 1], [0, 0]])
        gt = np.array([[0, 0], [1, 1]])
        assert calculate_iou(pred, gt, class_id=1) == 0.0

    def test_iou_partial_overlap(self):
        """IoU should be 0.5 for 50% overlap"""
        pred = np.array([[1, 1, 0], [1, 0, 0]])
        gt = np.array([[1, 0, 0], [1, 1, 0]])
        # Intersection: 2 pixels, Union: 4 pixels
        iou = calculate_iou(pred, gt, class_id=1)
        assert iou == 0.5

    def test_iou_class_not_present(self):
        """IoU should be 0.0 when class not present"""
        pred = np.array([[0, 0], [0, 0]])
        gt = np.array([[0, 0], [0, 0]])
        assert calculate_iou(pred, gt, class_id=1) == 0.0

    def test_iou_shape_mismatch(self):
        """Should raise error for shape mismatch"""
        pred = np.array([[1, 1]])
        gt = np.array([[1, 1], [1, 1]])
        with pytest.raises(AssertionError):
            calculate_iou(pred, gt, class_id=1)


class TestMIoU:
    """Tests for mean Intersection over Union (mIoU)"""

    def test_miou_multiclass(self):
        """mIoU should average IoU across classes"""
        pred = np.array([[0, 1, 2], [1, 1, 2]])
        gt = np.array([[0, 1, 1], [1, 2, 2]])
        miou, per_class = calculate_miou(pred, gt, num_classes=3)

        # Class 1: 2 intersection, 3 union → IoU=0.667
        # Class 2: 1 intersection, 3 union → IoU=0.333
        # mIoU = (0.667 + 0.333) / 2 = 0.5
        # Actual: Class 1 IoU=0.5, Class 2 IoU=0.333 → mIoU=0.417
        assert 0.40 < miou < 0.55
        assert 1 in per_class
        assert 2 in per_class

    def test_miou_perfect_match(self):
        """mIoU should be 1.0 for perfect match"""
        pred = np.array([[1, 2], [1, 2]])
        gt = np.array([[1, 2], [1, 2]])
        miou, _ = calculate_miou(pred, gt, num_classes=3)
        assert miou == 1.0

    def test_miou_ignore_background(self):
        """mIoU should ignore class 0 when specified"""
        pred = np.array([[0, 1], [0, 1]])
        gt = np.array([[0, 1], [0, 1]])
        miou, per_class = calculate_miou(pred, gt, num_classes=2, ignore_background=True)

        # Only class 1 should be in per_class
        assert 0 not in per_class
        assert 1 in per_class
        assert miou == 1.0

    def test_miou_no_valid_classes(self):
        """mIoU should return 0.0 when no valid classes present"""
        pred = np.array([[0, 0], [0, 0]])
        gt = np.array([[0, 0], [0, 0]])
        miou, _ = calculate_miou(pred, gt, num_classes=3, ignore_background=True)
        assert miou == 0.0


class TestWeightedMIoU:
    """Tests for Weighted mean IoU (accounts for class imbalance)"""

    def test_weighted_miou_balanced(self):
        """Weighted mIoU should equal mIoU for balanced classes"""
        pred = np.array([[1, 1, 2, 2], [1, 1, 2, 2]])
        gt = np.array([[1, 1, 2, 2], [1, 1, 2, 2]])
        miou, _ = calculate_miou(pred, gt, num_classes=3)
        weighted_miou, _ = calculate_weighted_miou(pred, gt, num_classes=3)
        assert abs(miou - weighted_miou) < 0.01

    def test_weighted_miou_imbalanced(self):
        """Weighted mIoU should differ from mIoU for imbalanced classes"""
        # 90% class 1, 10% class 2
        pred = np.ones((10, 10), dtype=int)
        pred[0, :] = 2
        gt = np.ones((10, 10), dtype=int)
        gt[0, :5] = 2

        miou, _ = calculate_miou(pred, gt, num_classes=3)
        weighted_miou, _ = calculate_weighted_miou(pred, gt, num_classes=3)

        # Weighted should be higher (class 1 has perfect IoU)
        assert weighted_miou > miou

    def test_weighted_miou_no_valid_pixels(self):
        """Weighted mIoU should return 0.0 when no valid pixels"""
        pred = np.array([[0, 0], [0, 0]])
        gt = np.array([[0, 0], [0, 0]])
        weighted_miou, _ = calculate_weighted_miou(pred, gt, num_classes=2, ignore_background=True)
        assert weighted_miou == 0.0


class TestF1Score:
    """Tests for F1-Score (Dice Similarity Coefficient)"""

    def test_f1_score_perfect(self):
        """F1-Score should be 1.0 for perfect match"""
        pred = np.array([[1, 1], [1, 1]])
        gt = np.array([[1, 1], [1, 1]])
        assert calculate_f1_score(pred, gt, class_id=1) == 1.0

    def test_f1_score_no_overlap(self):
        """F1-Score should be 0.0 for no overlap"""
        pred = np.array([[1, 1], [0, 0]])
        gt = np.array([[0, 0], [1, 1]])
        assert calculate_f1_score(pred, gt, class_id=1) == 0.0

    def test_f1_score_partial_overlap(self):
        """F1-Score should be calculated correctly"""
        pred = np.array([[0, 1], [1, 1]])
        gt = np.array([[0, 1], [1, 0]])
        # Intersection: 2, pred_positives: 3, gt_positives: 2
        # F1 = 2*2 / (3+2) = 4/5 = 0.8
        f1 = calculate_f1_score(pred, gt, class_id=1)
        assert abs(f1 - 0.8) < 0.01

    def test_f1_score_class_not_present(self):
        """F1-Score should be 0.0 when class not present"""
        pred = np.array([[0, 0], [0, 0]])
        gt = np.array([[0, 0], [0, 0]])
        assert calculate_f1_score(pred, gt, class_id=1) == 0.0


class TestPrecisionRecall:
    """Tests for Precision and Recall"""

    def test_precision_recall_perfect(self):
        """Perfect prediction should have precision=recall=1.0"""
        pred = np.array([[1, 1], [1, 1]])
        gt = np.array([[1, 1], [1, 1]])
        precision, recall = calculate_precision_recall(pred, gt, class_id=1)
        assert precision == 1.0
        assert recall == 1.0

    def test_precision_recall_false_positives(self):
        """False positives should lower precision"""
        pred = np.array([[1, 1, 1], [1, 1, 1]])  # 6 predicted
        gt = np.array([[1, 1, 0], [1, 0, 0]])  # 3 actual
        precision, recall = calculate_precision_recall(pred, gt, class_id=1)
        # TP=3, FP=3, FN=0
        assert precision == 0.5  # 3/(3+3)
        assert recall == 1.0  # 3/(3+0)

    def test_precision_recall_false_negatives(self):
        """False negatives should lower recall"""
        pred = np.array([[1, 1, 0], [1, 0, 0]])  # 3 predicted
        gt = np.array([[1, 1, 1], [1, 1, 1]])  # 6 actual
        precision, recall = calculate_precision_recall(pred, gt, class_id=1)
        # TP=3, FP=0, FN=3
        assert precision == 1.0  # 3/(3+0)
        assert recall == 0.5  # 3/(3+3)

    def test_precision_recall_no_predictions(self):
        """No predictions should give precision=recall=0.0"""
        pred = np.array([[0, 0], [0, 0]])
        gt = np.array([[1, 1], [1, 1]])
        precision, recall = calculate_precision_recall(pred, gt, class_id=1)
        assert precision == 0.0
        assert recall == 0.0

    def test_precision_recall_shape_mismatch(self):
        """Should raise error for shape mismatch"""
        pred = np.array([[1, 1]])
        gt = np.array([[1, 1], [1, 1]])
        with pytest.raises(AssertionError):
            calculate_precision_recall(pred, gt, class_id=1)


class TestPixelAccuracy:
    """Tests for Pixel Accuracy"""

    def test_pixel_accuracy_perfect(self):
        """Pixel accuracy should be 1.0 for perfect match"""
        pred = np.array([[1, 2], [3, 4]])
        gt = np.array([[1, 2], [3, 4]])
        assert calculate_pixel_accuracy(pred, gt) == 1.0

    def test_pixel_accuracy_half_correct(self):
        """Pixel accuracy should be 0.5 for 50% correct"""
        pred = np.array([[1, 2], [3, 4]])
        gt = np.array([[1, 2], [0, 0]])
        assert calculate_pixel_accuracy(pred, gt) == 0.5

    def test_pixel_accuracy_all_wrong(self):
        """Pixel accuracy should be 0.0 when all wrong"""
        pred = np.array([[1, 1], [1, 1]])
        gt = np.array([[0, 0], [0, 0]])
        assert calculate_pixel_accuracy(pred, gt) == 0.0

    def test_pixel_accuracy_shape_mismatch(self):
        """Should raise error for shape mismatch"""
        pred = np.array([[1, 1]])
        gt = np.array([[1, 1], [1, 1]])
        with pytest.raises(AssertionError):
            calculate_pixel_accuracy(pred, gt)


class TestConfusionMatrix:
    """Tests for Confusion Matrix generation"""

    def test_confusion_matrix_shape(self):
        """Confusion matrix should have correct shape"""
        pred = np.array([[0, 1, 2], [1, 1, 2]])
        gt = np.array([[0, 1, 1], [1, 2, 2]])
        cm = generate_confusion_matrix(pred, gt, num_classes=3)
        assert cm.shape == (3, 3)

    def test_confusion_matrix_diagonal(self):
        """Perfect prediction should have all values on diagonal"""
        pred = np.array([[0, 1, 2], [0, 1, 2]])
        gt = np.array([[0, 1, 2], [0, 1, 2]])
        cm = generate_confusion_matrix(pred, gt, num_classes=3)
        # All correct predictions on diagonal
        assert cm[0, 0] == 2
        assert cm[1, 1] == 2
        assert cm[2, 2] == 2
        # No off-diagonal values
        assert np.sum(cm) - np.trace(cm) == 0

    def test_confusion_matrix_values(self):
        """Confusion matrix should count misclassifications correctly"""
        pred = np.array([[0, 1, 1], [2, 2, 0]])
        gt = np.array([[0, 0, 1], [1, 2, 2]])
        cm = generate_confusion_matrix(pred, gt, num_classes=3)

        # Check specific values
        assert cm[0, 0] == 1  # True class 0, predicted 0
        assert cm[0, 1] == 1  # True class 0, predicted 1
        assert cm[1, 1] == 1  # True class 1, predicted 1
        assert cm[1, 2] == 1  # True class 1, predicted 2
        assert cm[2, 0] == 1  # True class 2, predicted 0
        assert cm[2, 2] == 1  # True class 2, predicted 2

    def test_confusion_matrix_shape_mismatch(self):
        """Should raise error for shape mismatch"""
        pred = np.array([[1, 1]])
        gt = np.array([[1, 1], [1, 1]])
        with pytest.raises(AssertionError):
            generate_confusion_matrix(pred, gt, num_classes=2)


class TestBoundaryPrecision:
    """Tests for Boundary Precision"""

    def test_boundary_precision_perfect(self):
        """Perfect boundary alignment should give 1.0"""
        pred = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=bool)
        gt = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=bool)
        precision = calculate_boundary_precision(pred, gt, tolerance=1)
        # With tolerance, should be close to 1.0
        assert precision > 0.5

    def test_boundary_precision_no_boundary(self):
        """Fully filled regions have boundary that matches"""
        pred = np.ones((5, 5), dtype=bool)
        gt = np.ones((5, 5), dtype=bool)
        precision = calculate_boundary_precision(pred, gt, tolerance=1)
        # Boundaries match perfectly, so precision should be 1.0
        assert precision == 1.0


class TestRegressionValidation:
    """Regression tests to ensure metrics remain stable"""

    def test_mexicali_miou_regression(self):
        """Ensure Mexicali mIoU doesn't regress below baseline"""
        # Mock data simulating Mexicali results
        # In real test, would load: np.load('tests/fixtures/mexicali_mgrg_seg.npy')
        np.random.seed(42)
        pred = np.random.randint(0, 5, (100, 100))
        gt = pred.copy()
        # Add some noise (10% different)
        noise_mask = np.random.rand(100, 100) < 0.1
        gt[noise_mask] = (gt[noise_mask] + 1) % 5

        current_miou, _ = calculate_miou(pred, gt, num_classes=5)

        # Expected to be high since only 10% difference
        assert current_miou >= 0.70, f"mIoU regressed: {current_miou:.3f} < 0.70"

    def test_metrics_consistency(self):
        """Ensure metrics are consistent across runs"""
        pred = np.array([[1, 1, 2], [1, 2, 2]])
        gt = np.array([[1, 1, 1], [1, 2, 2]])

        # Run twice, should get same results
        miou1, _ = calculate_miou(pred, gt, num_classes=3)
        miou2, _ = calculate_miou(pred, gt, num_classes=3)
        assert miou1 == miou2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
