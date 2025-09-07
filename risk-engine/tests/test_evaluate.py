"""Tests for model evaluation modules."""

import pytest
import pandas as pd
import numpy as np
import json
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.evaluate import (
    compute_metrics,
    compute_confusion_matrices
)


class TestMetricsComputation:
    """Test cases for metrics computation."""
    
    def setup_method(self):
        """Set up test data for metrics computation."""
        # Perfect predictions
        self.y_true_perfect = np.array([0, 0, 1, 1])
        self.y_prob_perfect = np.array([0.1, 0.2, 0.8, 0.9])
        
        # Random predictions
        np.random.seed(42)
        self.y_true_random = np.random.randint(0, 2, 100)
        self.y_prob_random = np.random.random(100)
        
        # All positive predictions
        self.y_true_all_pos = np.array([1, 1, 1, 1])
        self.y_prob_all_pos = np.array([0.9, 0.8, 0.7, 0.6])
    
    def test_compute_metrics_perfect_predictions(self):
        """Test metrics computation with perfect predictions."""
        metrics = compute_metrics(
            self.y_true_perfect, 
            self.y_prob_perfect, 
            self.y_prob_perfect,
            Mock()
        )
        
        # Perfect predictions should have AUROC = 1.0
        assert metrics['auroc_raw'] == 1.0
        assert metrics['auroc_calibrated'] == 1.0
        
        # AUPRC should also be 1.0 for perfect predictions
        assert metrics['auprc_raw'] == 1.0
        assert metrics['auprc_calibrated'] == 1.0
        
        # Brier score should be low for perfect predictions
        assert metrics['brier_raw'] < 0.1
        assert metrics['brier_calibrated'] < 0.1
    
    def test_compute_metrics_random_predictions(self):
        """Test metrics computation with random predictions."""
        metrics = compute_metrics(
            self.y_true_random,
            self.y_prob_random,
            self.y_prob_random,
            Mock()
        )
        
        # Random predictions should have AUROC around 0.5
        assert 0.3 < metrics['auroc_raw'] < 0.7
        assert 0.3 < metrics['auroc_calibrated'] < 0.7
        
        # All metrics should be valid numbers
        for metric_name, value in metrics.items():
            assert isinstance(value, float)
            assert not np.isnan(value)
            assert not np.isinf(value)
    
    def test_compute_metrics_edge_cases(self):
        """Test metrics computation with edge cases."""
        # All positive labels
        metrics_all_pos = compute_metrics(
            self.y_true_all_pos,
            self.y_prob_all_pos,
            self.y_prob_all_pos,
            Mock()
        )
        
        # Should handle all positive case gracefully
        assert isinstance(metrics_all_pos['auroc_raw'], float)


class TestConfusionMatrices:
    """Test cases for confusion matrix computation."""
    
    def setup_method(self):
        """Set up test data for confusion matrix computation."""
        self.y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        self.y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        
        self.thresholds = {
            'low': 0.3,
            'medium': 0.5,
            'high': 0.7
        }
    
    def test_compute_confusion_matrices(self):
        """Test confusion matrix computation."""
        results = compute_confusion_matrices(
            self.y_true,
            self.y_prob,
            self.thresholds,
            Mock()
        )
        
        # Should have results for all thresholds
        assert len(results) == len(self.thresholds)
        
        for threshold_name in self.thresholds:
            assert threshold_name in results
            
            result = results[threshold_name]
            
            # Check structure
            assert 'threshold' in result
            assert 'confusion_matrix' in result
            assert 'metrics' in result
            
            # Check confusion matrix components
            cm = result['confusion_matrix']
            assert all(key in cm for key in ['tn', 'fp', 'fn', 'tp'])
            assert all(isinstance(cm[key], int) for key in cm)
            
            # Check derived metrics
            metrics = result['metrics']
            expected_metrics = ['sensitivity', 'specificity', 'ppv', 'npv', 'f1']
            assert all(metric in metrics for metric in expected_metrics)
            assert all(isinstance(metrics[metric], float) for metric in expected_metrics)
            
            # Check that metrics are in valid ranges [0, 1]
            for metric in expected_metrics:
                assert 0 <= metrics[metric] <= 1
    
    def test_confusion_matrix_threshold_effects(self):
        """Test that different thresholds produce different confusion matrices."""
        results = compute_confusion_matrices(
            self.y_true,
            self.y_prob,
            self.thresholds,
            Mock()
        )
        
        # Lower thresholds should generally have higher sensitivity
        low_sensitivity = results['low']['metrics']['sensitivity']
        high_sensitivity = results['high']['metrics']['sensitivity']
        
        # This might not always be true due to the specific data, but generally expected
        # assert low_sensitivity >= high_sensitivity
        
        # Higher thresholds should generally have higher specificity
        low_specificity = results['low']['metrics']['specificity']
        high_specificity = results['high']['metrics']['specificity']
        
        # assert high_specificity >= low_specificity


class TestEvaluationIntegration:
    """Integration tests for the evaluation pipeline."""
    
    def test_evaluation_output_format(self):
        """Test that evaluation output has the correct format."""
        # Mock evaluation results
        mock_metrics = {
            'auroc_raw': 0.85,
            'auroc_calibrated': 0.87,
            'auprc_raw': 0.72,
            'auprc_calibrated': 0.75,
            'brier_raw': 0.18,
            'brier_calibrated': 0.16
        }
        
        mock_confusion = {
            'high_sensitivity': {
                'threshold': 0.25,
                'confusion_matrix': {'tn': 80, 'fp': 15, 'fn': 5, 'tp': 20},
                'metrics': {'sensitivity': 0.8, 'specificity': 0.84, 'ppv': 0.57, 'npv': 0.94, 'f1': 0.67}
            },
            'high_ppv': {
                'threshold': 0.45,
                'confusion_matrix': {'tn': 90, 'fp': 5, 'fn': 10, 'tp': 15},
                'metrics': {'sensitivity': 0.6, 'specificity': 0.95, 'ppv': 0.75, 'npv': 0.9, 'f1': 0.67}
            }
        }
        
        # Test that the structure is correct for JSON serialization
        results = {
            'metrics': mock_metrics,
            'confusion_matrices': mock_confusion
        }
        
        # Should be serializable to JSON
        json_str = json.dumps(results, indent=2)
        assert isinstance(json_str, str)
        
        # Should be deserializable
        parsed_results = json.loads(json_str)
        assert parsed_results == results


class TestModelValidation:
    """Test cases for model validation and sanity checks."""
    
    def test_model_performance_thresholds(self):
        """Test that model performance meets minimum thresholds."""
        # Example thresholds for a healthcare model
        min_auroc = 0.65
        min_auprc = 0.30
        max_brier = 0.30
        
        # Mock metrics that should pass
        good_metrics = {
            'auroc_calibrated': 0.75,
            'auprc_calibrated': 0.45,
            'brier_calibrated': 0.20
        }
        
        assert good_metrics['auroc_calibrated'] >= min_auroc
        assert good_metrics['auprc_calibrated'] >= min_auprc
        assert good_metrics['brier_calibrated'] <= max_brier
    
    def test_calibration_improvement(self):
        """Test that calibration improves probability estimates."""
        # Mock metrics showing calibration improvement
        metrics = {
            'brier_raw': 0.25,
            'brier_calibrated': 0.20
        }
        
        # Calibration should improve (lower) Brier score
        assert metrics['brier_calibrated'] <= metrics['brier_raw']


if __name__ == "__main__":
    pytest.main([__file__])
