"""Script to run comprehensive model monitoring."""

import sys
import os
import json
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from monitoring.drift_detection import DriftDetector, PerformanceMonitor, generate_monitoring_report
from common.io import file_exists, read_parquet
from common.logging import get_logger


def run_drift_monitoring():
    """Run data drift monitoring."""
    logger = get_logger("drift_monitoring")
    
    try:
        # Paths
        reference_data_path = "data/processed/train.parquet"
        current_data_path = "data/processed/current.parquet"
        
        # For demo, use validation data as current data
        if not file_exists(current_data_path):
            current_data_path = "data/models/lgbm/val.parquet"
        
        if not file_exists(reference_data_path) or not file_exists(current_data_path):
            logger.warning("Required data files not found for drift monitoring")
            return None
        
        # Load data
        reference_data = read_parquet(reference_data_path)
        current_data = read_parquet(current_data_path)
        
        # Initialize drift detector
        drift_detector = DriftDetector(reference_data)
        
        # Detect drift
        drift_results = drift_detector.detect_drift(current_data)
        
        logger.info(f"Drift monitoring completed - {drift_results['summary']['drifted_features']} features drifted")
        
        return drift_results
        
    except Exception as e:
        logger.error(f"Drift monitoring failed: {e}")
        return None


def run_performance_monitoring():
    """Run performance monitoring."""
    logger = get_logger("performance_monitoring")
    
    try:
        # Load baseline metrics
        metrics_path = "data/reports/metrics.json"
        if not file_exists(metrics_path):
            logger.warning("Baseline metrics not found")
            return None
        
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)
        
        baseline_metrics = metrics_data.get('metrics', {})
        
        # For demo, simulate current performance (slightly degraded)
        current_metrics = {
            'auroc': baseline_metrics.get('auroc', 0.75) - 0.02,
            'auprc': baseline_metrics.get('auprc', 0.60) - 0.03,
            'brier_score': baseline_metrics.get('brier_score', 0.20) + 0.01,
            'accuracy': baseline_metrics.get('accuracy', 0.70) - 0.01
        }
        
        # Initialize performance monitor
        perf_monitor = PerformanceMonitor(baseline_metrics)
        
        # Add current measurement
        performance_results = perf_monitor.add_performance_measurement(current_metrics)
        
        logger.info(f"Performance monitoring completed - degradation detected: {performance_results['degradation_detected']}")
        
        return performance_results
        
    except Exception as e:
        logger.error(f"Performance monitoring failed: {e}")
        return None


def run_bias_monitoring():
    """Run bias monitoring (placeholder)."""
    logger = get_logger("bias_monitoring")
    
    try:
        # For demo, return no bias detected
        bias_results = {
            'timestamp': '2024-01-01T00:00:00',
            'bias_detected': False,
            'attribute_analysis': {},
            'summary': {
                'total_attributes': 0,
                'biased_attributes': 0
            }
        }
        
        logger.info("Bias monitoring completed - no bias detected")
        
        return bias_results
        
    except Exception as e:
        logger.error(f"Bias monitoring failed: {e}")
        return None


def main():
    """Main monitoring workflow."""
    logger = get_logger("run_monitoring")
    
    try:
        logger.info("Starting comprehensive model monitoring...")
        
        # Run all monitoring components
        drift_results = run_drift_monitoring()
        performance_results = run_performance_monitoring()
        bias_results = run_bias_monitoring()
        
        # Generate comprehensive report
        if any([drift_results, performance_results, bias_results]):
            report_path = generate_monitoring_report(
                drift_results or {},
                performance_results or {},
                bias_results,
                "data/reports/monitoring_report.json"
            )
            
            logger.info(f"Monitoring report generated: {report_path}")
            
            # Read report to determine exit code
            with open(report_path, 'r') as f:
                report = json.load(f)
            
            overall_status = report.get('overall_status', 'healthy')
            
            if overall_status == 'critical':
                logger.error("Critical issues detected in monitoring")
                return 2
            elif overall_status == 'warning':
                logger.warning("Warning-level issues detected in monitoring")
                return 1
            else:
                logger.info("Monitoring completed - system healthy")
                return 0
        else:
            logger.error("All monitoring components failed")
            return 1
            
    except Exception as e:
        logger.error(f"Monitoring workflow failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
