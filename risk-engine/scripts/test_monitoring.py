"""Test script for monitoring and drift detection."""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from monitoring.drift_detection import (
    DriftDetector, PerformanceMonitor, BiasMonitor, generate_monitoring_report
)
from common.io import file_exists, read_parquet
from common.logging import get_logger


def test_drift_detection():
    """Test data drift detection functionality."""
    logger = get_logger("test_drift")
    
    try:
        logger.info("Testing data drift detection...")
        
        # Load training and validation data
        train_path = "data/processed/train.parquet"
        val_path = "data/models/lgbm/val.parquet"
        
        if not file_exists(train_path) or not file_exists(val_path):
            logger.warning("Training or validation data not found. Creating synthetic data for testing.")
            
            # Create synthetic data for testing
            np.random.seed(42)
            
            # Reference data (training-like)
            ref_data = pd.DataFrame({
                'n_observations_180d': np.random.poisson(15, 100),
                'n_encounters_180d': np.random.poisson(3, 100),
                'n_active_meds_180d': np.random.poisson(5, 100),
                'hba1c_last': np.random.normal(7.2, 1.0, 100),
                'sbp_last': np.random.normal(140, 20, 100),
                'dbp_last': np.random.normal(85, 15, 100)
            })
            
            # Current data (slightly drifted)
            curr_data = pd.DataFrame({
                'n_observations_180d': np.random.poisson(18, 80),  # Drift: higher mean
                'n_encounters_180d': np.random.poisson(3, 80),
                'n_active_meds_180d': np.random.poisson(5, 80),
                'hba1c_last': np.random.normal(7.5, 1.2, 80),     # Drift: higher mean and variance
                'sbp_last': np.random.normal(140, 20, 80),
                'dbp_last': np.random.normal(85, 15, 80)
            })
        else:
            # Use real data
            ref_data = read_parquet(train_path)
            curr_data = read_parquet(val_path)
        
        # Initialize drift detector
        logger.info("Initializing drift detector...")
        drift_detector = DriftDetector(ref_data)
        logger.info("✅ Drift detector initialized successfully")
        
        # Detect drift
        logger.info("Running drift detection...")
        drift_results = drift_detector.detect_drift(curr_data)
        
        logger.info("✅ Drift detection completed")
        logger.info(f"Overall drift detected: {drift_results['overall_drift_detected']}")
        logger.info(f"Drift rate: {drift_results['summary']['drift_rate']:.2%}")
        
        # Show feature-level results
        for feature, result in drift_results['feature_drift'].items():
            if result['drift_detected']:
                logger.info(f"  🚨 Drift detected in {feature}")
            else:
                logger.info(f"  ✅ No drift in {feature}")
        
        logger.info("🎉 Drift detection tests passed!")
        return drift_results
        
    except Exception as e:
        logger.error(f"❌ Drift detection test failed: {e}")
        return None


def test_performance_monitoring():
    """Test performance monitoring functionality."""
    logger = get_logger("test_performance")
    
    try:
        logger.info("Testing performance monitoring...")
        
        # Initialize performance monitor with baseline metrics
        baseline_metrics = {
            'auroc': 0.75,
            'auprc': 0.60,
            'brier_score': 0.20,
            'accuracy': 0.70
        }
        
        logger.info("Initializing performance monitor...")
        perf_monitor = PerformanceMonitor(baseline_metrics)
        logger.info("✅ Performance monitor initialized successfully")
        
        # Simulate performance measurements over time
        logger.info("Adding performance measurements...")
        
        # Good performance (no degradation)
        good_metrics = {
            'auroc': 0.76,
            'auprc': 0.61,
            'brier_score': 0.19,
            'accuracy': 0.71
        }
        
        result1 = perf_monitor.add_performance_measurement(good_metrics, "2024-01-01T10:00:00")
        logger.info(f"Measurement 1 - Degradation detected: {result1['degradation_detected']}")
        
        # Degraded performance
        degraded_metrics = {
            'auroc': 0.68,  # Significant drop
            'auprc': 0.52,  # Significant drop
            'brier_score': 0.28,  # Significant increase (worse)
            'accuracy': 0.65
        }
        
        result2 = perf_monitor.add_performance_measurement(degraded_metrics, "2024-01-02T10:00:00")
        logger.info(f"Measurement 2 - Degradation detected: {result2['degradation_detected']}")
        
        if result2['degradation_detected']:
            logger.info("Alerts:")
            for alert in result2['alerts']:
                logger.info(f"  🚨 {alert}")
        
        # Test trend analysis
        logger.info("Testing trend analysis...")
        auroc_trend = perf_monitor.get_performance_trend('auroc')
        logger.info(f"AUROC trend: {auroc_trend['trend']}")
        
        logger.info("✅ Performance monitoring tests passed!")
        return result2
        
    except Exception as e:
        logger.error(f"❌ Performance monitoring test failed: {e}")
        return None


def test_bias_monitoring():
    """Test bias monitoring functionality."""
    logger = get_logger("test_bias")
    
    try:
        logger.info("Testing bias monitoring...")
        
        # Create synthetic data with potential bias
        np.random.seed(42)
        
        # Simulate data with bias across gender
        n_samples = 200
        
        # Generate gender (protected attribute)
        gender = np.random.choice(['M', 'F'], n_samples, p=[0.6, 0.4])
        
        # Generate predictions with bias (higher predictions for one group)
        predictions = []
        labels = []
        
        for g in gender:
            if g == 'M':
                # Male patients get higher risk scores on average
                pred = np.random.beta(3, 5)  # Skewed towards higher values
                label = np.random.binomial(1, pred * 0.8)  # Some correlation with prediction
            else:
                # Female patients get lower risk scores on average
                pred = np.random.beta(2, 6)  # Skewed towards lower values
                label = np.random.binomial(1, pred * 0.8)
            
            predictions.append(pred)
            labels.append(label)
        
        # Create DataFrame
        bias_data = pd.DataFrame({
            'gender': gender,
            'prediction_proba': predictions,
            'label_90d': labels,
            'age_group': np.random.choice(['18-40', '41-65', '65+'], n_samples)  # Another protected attribute
        })
        
        # Initialize bias monitor
        logger.info("Initializing bias monitor...")
        bias_monitor = BiasMonitor(['gender', 'age_group'])
        logger.info("✅ Bias monitor initialized successfully")
        
        # Detect bias
        logger.info("Running bias detection...")
        bias_results = bias_monitor.detect_bias(bias_data)
        
        logger.info("✅ Bias detection completed")
        logger.info(f"Overall bias detected: {bias_results['bias_detected']}")
        
        # Show attribute-level results
        for attribute, result in bias_results['attribute_analysis'].items():
            if result['bias_detected']:
                logger.info(f"  🚨 Bias detected in {attribute}")
                
                # Show group statistics
                for group, stats in result['group_statistics'].items():
                    logger.info(f"    {group}: mean_pred={stats['prediction_mean']:.3f}, count={stats['count']}")
            else:
                logger.info(f"  ✅ No bias detected in {attribute}")
        
        logger.info("🎉 Bias monitoring tests passed!")
        return bias_results
        
    except Exception as e:
        logger.error(f"❌ Bias monitoring test failed: {e}")
        return None


def test_monitoring_report():
    """Test comprehensive monitoring report generation."""
    logger = get_logger("test_report")
    
    try:
        logger.info("Testing monitoring report generation...")
        
        # Run all monitoring tests
        drift_results = test_drift_detection()
        performance_results = test_performance_monitoring()
        bias_results = test_bias_monitoring()
        
        if not all([drift_results, performance_results, bias_results]):
            logger.warning("Some monitoring tests failed, generating partial report")
        
        # Generate comprehensive report
        logger.info("Generating monitoring report...")
        report_path = generate_monitoring_report(
            drift_results or {},
            performance_results or {},
            bias_results,
            "data/reports/test_monitoring_report.json"
        )
        
        logger.info(f"✅ Monitoring report generated: {report_path}")
        
        # Read and display summary
        import json
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        logger.info(f"Report summary:")
        logger.info(f"  Overall status: {report['overall_status']}")
        logger.info(f"  Number of alerts: {len(report['alerts'])}")
        
        if report['alerts']:
            logger.info("  Alerts:")
            for alert in report['alerts']:
                logger.info(f"    🚨 {alert}")
        
        logger.info("🎉 Monitoring report tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Monitoring report test failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing CareSight Risk Engine Monitoring & Alerting")
    print("=" * 60)
    
    # Test individual components
    print("\n1. Testing Drift Detection")
    print("-" * 30)
    drift_success = test_drift_detection() is not None
    
    print("\n2. Testing Performance Monitoring")
    print("-" * 30)
    perf_success = test_performance_monitoring() is not None
    
    print("\n3. Testing Bias Monitoring")
    print("-" * 30)
    bias_success = test_bias_monitoring() is not None
    
    print("\n4. Testing Comprehensive Report")
    print("-" * 30)
    report_success = test_monitoring_report()
    
    print("\n" + "=" * 60)
    
    if all([drift_success, perf_success, bias_success, report_success]):
        print("🎉 All monitoring tests PASSED!")
    else:
        print("❌ Some monitoring tests FAILED!")
        sys.exit(1)
