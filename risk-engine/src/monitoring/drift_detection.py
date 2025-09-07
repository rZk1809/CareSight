"""Data drift detection and monitoring for the CareSight Risk Engine.

This module implements statistical tests and monitoring capabilities to detect
data drift, model performance degradation, and bias in predictions.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from common.io import read_parquet, write_parquet, ensure_dir, file_exists
from common.logging import get_logger

# Import statistical tests
try:
    from scipy import stats
    from scipy.stats import ks_2samp, chi2_contingency
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy not available. Install with: pip install scipy")


class DriftDetector:
    """Detects data drift using statistical tests."""
    
    def __init__(self, reference_data: pd.DataFrame, feature_columns: List[str] = None):
        """Initialize drift detector.
        
        Args:
            reference_data: Reference dataset (e.g., training data)
            feature_columns: List of feature columns to monitor
        """
        self.logger = get_logger("drift_detector")
        self.reference_data = reference_data
        
        if feature_columns is None:
            # Auto-detect feature columns (exclude common ID/target columns)
            exclude_cols = ['patient', 'as_of', 'label_90d', 'prediction_proba', 'prediction']
            self.feature_columns = [col for col in reference_data.columns 
                                  if col not in exclude_cols and 
                                  reference_data[col].dtype in ['int64', 'float64']]
        else:
            self.feature_columns = feature_columns
        
        self.logger.info(f"Initialized drift detector with {len(self.feature_columns)} features")
        self.logger.info(f"Features: {self.feature_columns}")
        
        # Compute reference statistics
        self._compute_reference_stats()
    
    def _compute_reference_stats(self):
        """Compute reference statistics for drift detection."""
        self.reference_stats = {}
        
        for feature in self.feature_columns:
            if feature in self.reference_data.columns:
                values = self.reference_data[feature].dropna()
                
                self.reference_stats[feature] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'median': float(values.median()),
                    'q25': float(values.quantile(0.25)),
                    'q75': float(values.quantile(0.75)),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'count': len(values),
                    'missing_rate': (self.reference_data[feature].isna().sum() / len(self.reference_data))
                }
        
        self.logger.info("Reference statistics computed")
    
    def detect_drift(self, current_data: pd.DataFrame, 
                    significance_level: float = 0.05) -> Dict[str, Any]:
        """Detect drift between reference and current data.
        
        Args:
            current_data: Current dataset to compare against reference
            significance_level: Significance level for statistical tests
            
        Returns:
            Dictionary with drift detection results
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy required for drift detection")
        
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'significance_level': significance_level,
            'overall_drift_detected': False,
            'feature_drift': {},
            'summary': {
                'total_features': len(self.feature_columns),
                'drifted_features': 0,
                'drift_rate': 0.0
            }
        }
        
        drifted_features = 0
        
        for feature in self.feature_columns:
            if feature not in current_data.columns:
                self.logger.warning(f"Feature {feature} not found in current data")
                continue
            
            # Get clean data
            ref_values = self.reference_data[feature].dropna()
            curr_values = current_data[feature].dropna()
            
            if len(ref_values) == 0 or len(curr_values) == 0:
                self.logger.warning(f"No valid values for feature {feature}")
                continue
            
            # Perform statistical tests
            feature_result = self._test_feature_drift(
                ref_values, curr_values, feature, significance_level
            )
            
            drift_results['feature_drift'][feature] = feature_result
            
            if feature_result['drift_detected']:
                drifted_features += 1
        
        # Update summary
        drift_results['summary']['drifted_features'] = drifted_features
        drift_results['summary']['drift_rate'] = drifted_features / len(self.feature_columns)
        drift_results['overall_drift_detected'] = drift_results['summary']['drift_rate'] > 0.1  # 10% threshold
        
        self.logger.info(f"Drift detection completed: {drifted_features}/{len(self.feature_columns)} features drifted")
        
        return drift_results
    
    def _test_feature_drift(self, ref_values: pd.Series, curr_values: pd.Series, 
                           feature_name: str, significance_level: float) -> Dict[str, Any]:
        """Test drift for a single feature.
        
        Args:
            ref_values: Reference values
            curr_values: Current values
            feature_name: Name of the feature
            significance_level: Significance level
            
        Returns:
            Dictionary with test results
        """
        result = {
            'feature': feature_name,
            'drift_detected': False,
            'tests': {},
            'statistics': {}
        }
        
        # Compute current statistics
        result['statistics'] = {
            'reference': {
                'mean': float(ref_values.mean()),
                'std': float(ref_values.std()),
                'count': len(ref_values)
            },
            'current': {
                'mean': float(curr_values.mean()),
                'std': float(curr_values.std()),
                'count': len(curr_values)
            }
        }
        
        # Kolmogorov-Smirnov test (for continuous distributions)
        try:
            ks_stat, ks_pvalue = ks_2samp(ref_values, curr_values)
            result['tests']['kolmogorov_smirnov'] = {
                'statistic': float(ks_stat),
                'p_value': float(ks_pvalue),
                'drift_detected': ks_pvalue < significance_level
            }
        except Exception as e:
            self.logger.warning(f"KS test failed for {feature_name}: {e}")
        
        # Mann-Whitney U test (non-parametric)
        try:
            mw_stat, mw_pvalue = stats.mannwhitneyu(ref_values, curr_values, alternative='two-sided')
            result['tests']['mann_whitney'] = {
                'statistic': float(mw_stat),
                'p_value': float(mw_pvalue),
                'drift_detected': mw_pvalue < significance_level
            }
        except Exception as e:
            self.logger.warning(f"Mann-Whitney test failed for {feature_name}: {e}")
        
        # Population Stability Index (PSI)
        try:
            psi_score = self._calculate_psi(ref_values, curr_values)
            result['tests']['population_stability_index'] = {
                'score': float(psi_score),
                'drift_detected': psi_score > 0.1  # PSI > 0.1 indicates drift
            }
        except Exception as e:
            self.logger.warning(f"PSI calculation failed for {feature_name}: {e}")
        
        # Determine overall drift for this feature
        drift_indicators = [test_result.get('drift_detected', False) 
                          for test_result in result['tests'].values()]
        result['drift_detected'] = any(drift_indicators)
        
        return result
    
    def _calculate_psi(self, reference: pd.Series, current: pd.Series, 
                      bins: int = 10) -> float:
        """Calculate Population Stability Index (PSI).
        
        Args:
            reference: Reference data
            current: Current data
            bins: Number of bins for discretization
            
        Returns:
            PSI score
        """
        # Create bins based on reference data
        _, bin_edges = np.histogram(reference, bins=bins)
        
        # Ensure bins cover the full range
        bin_edges[0] = min(bin_edges[0], current.min())
        bin_edges[-1] = max(bin_edges[-1], current.max())
        
        # Calculate distributions
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        curr_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Convert to proportions
        ref_props = ref_counts / len(reference)
        curr_props = curr_counts / len(current)
        
        # Avoid division by zero
        ref_props = np.where(ref_props == 0, 0.0001, ref_props)
        curr_props = np.where(curr_props == 0, 0.0001, curr_props)
        
        # Calculate PSI
        psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
        
        return psi


class PerformanceMonitor:
    """Monitors model performance over time."""
    
    def __init__(self, baseline_metrics: Dict[str, float]):
        """Initialize performance monitor.
        
        Args:
            baseline_metrics: Baseline performance metrics
        """
        self.logger = get_logger("performance_monitor")
        self.baseline_metrics = baseline_metrics
        self.performance_history = []
        
        self.logger.info(f"Initialized performance monitor with baseline: {baseline_metrics}")
    
    def add_performance_measurement(self, metrics: Dict[str, float], 
                                  timestamp: str = None) -> Dict[str, Any]:
        """Add a new performance measurement.
        
        Args:
            metrics: Current performance metrics
            timestamp: Timestamp for the measurement
            
        Returns:
            Performance analysis results
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        measurement = {
            'timestamp': timestamp,
            'metrics': metrics.copy()
        }
        
        self.performance_history.append(measurement)
        
        # Analyze performance degradation
        analysis = self._analyze_performance_degradation(metrics)
        measurement['analysis'] = analysis
        
        self.logger.info(f"Added performance measurement: {len(self.performance_history)} total measurements")
        
        return analysis
    
    def _analyze_performance_degradation(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Analyze performance degradation compared to baseline.
        
        Args:
            current_metrics: Current performance metrics
            
        Returns:
            Analysis results
        """
        analysis = {
            'degradation_detected': False,
            'metric_changes': {},
            'alerts': []
        }
        
        # Define degradation thresholds
        degradation_thresholds = {
            'auroc': 0.05,      # 5% decrease
            'auprc': 0.05,      # 5% decrease
            'brier_score': 0.05  # 5% increase (worse)
        }
        
        for metric_name, current_value in current_metrics.items():
            if metric_name in self.baseline_metrics:
                baseline_value = self.baseline_metrics[metric_name]
                
                if metric_name == 'brier_score':
                    # For Brier score, lower is better
                    change = current_value - baseline_value
                    degraded = change > degradation_thresholds.get(metric_name, 0.05)
                else:
                    # For AUROC, AUPRC, higher is better
                    change = baseline_value - current_value
                    degraded = change > degradation_thresholds.get(metric_name, 0.05)
                
                analysis['metric_changes'][metric_name] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'change': change,
                    'degraded': degraded
                }
                
                if degraded:
                    analysis['degradation_detected'] = True
                    analysis['alerts'].append(
                        f"Performance degradation detected in {metric_name}: "
                        f"{baseline_value:.3f} → {current_value:.3f} (change: {change:+.3f})"
                    )
        
        return analysis
    
    def get_performance_trend(self, metric_name: str, window_size: int = 10) -> Dict[str, Any]:
        """Get performance trend for a specific metric.
        
        Args:
            metric_name: Name of the metric
            window_size: Size of the moving window
            
        Returns:
            Trend analysis
        """
        if len(self.performance_history) < 2:
            return {'trend': 'insufficient_data'}
        
        # Extract metric values
        values = []
        timestamps = []
        
        for measurement in self.performance_history:
            if metric_name in measurement['metrics']:
                values.append(measurement['metrics'][metric_name])
                timestamps.append(measurement['timestamp'])
        
        if len(values) < 2:
            return {'trend': 'insufficient_data'}
        
        # Calculate trend
        recent_values = values[-window_size:] if len(values) >= window_size else values
        
        if len(recent_values) >= 2:
            # Simple linear trend
            x = np.arange(len(recent_values))
            slope = np.polyfit(x, recent_values, 1)[0]
            
            if abs(slope) < 0.001:
                trend = 'stable'
            elif slope > 0:
                trend = 'improving' if metric_name != 'brier_score' else 'degrading'
            else:
                trend = 'degrading' if metric_name != 'brier_score' else 'improving'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'slope': float(slope) if 'slope' in locals() else 0.0,
            'recent_values': recent_values,
            'timestamps': timestamps[-len(recent_values):]
        }


class BiasMonitor:
    """Monitors model bias across different demographic groups."""
    
    def __init__(self, protected_attributes: List[str]):
        """Initialize bias monitor.
        
        Args:
            protected_attributes: List of protected attribute column names
        """
        self.logger = get_logger("bias_monitor")
        self.protected_attributes = protected_attributes
        
        self.logger.info(f"Initialized bias monitor for attributes: {protected_attributes}")
    
    def detect_bias(self, data: pd.DataFrame, prediction_col: str = 'prediction_proba',
                   target_col: str = 'label_90d') -> Dict[str, Any]:
        """Detect bias in model predictions.
        
        Args:
            data: Dataset with predictions and protected attributes
            prediction_col: Column name for predictions
            target_col: Column name for true labels
            
        Returns:
            Bias analysis results
        """
        bias_results = {
            'timestamp': datetime.now().isoformat(),
            'bias_detected': False,
            'attribute_analysis': {},
            'summary': {
                'total_attributes': len(self.protected_attributes),
                'biased_attributes': 0
            }
        }
        
        biased_attributes = 0
        
        for attribute in self.protected_attributes:
            if attribute not in data.columns:
                self.logger.warning(f"Protected attribute {attribute} not found in data")
                continue
            
            # Analyze bias for this attribute
            attribute_analysis = self._analyze_attribute_bias(
                data, attribute, prediction_col, target_col
            )
            
            bias_results['attribute_analysis'][attribute] = attribute_analysis
            
            if attribute_analysis['bias_detected']:
                biased_attributes += 1
        
        # Update summary
        bias_results['summary']['biased_attributes'] = biased_attributes
        bias_results['bias_detected'] = biased_attributes > 0
        
        self.logger.info(f"Bias analysis completed: {biased_attributes}/{len(self.protected_attributes)} attributes show bias")
        
        return bias_results
    
    def _analyze_attribute_bias(self, data: pd.DataFrame, attribute: str,
                               prediction_col: str, target_col: str) -> Dict[str, Any]:
        """Analyze bias for a single protected attribute.
        
        Args:
            data: Dataset
            attribute: Protected attribute name
            prediction_col: Prediction column
            target_col: Target column
            
        Returns:
            Bias analysis for the attribute
        """
        analysis = {
            'attribute': attribute,
            'bias_detected': False,
            'group_statistics': {},
            'fairness_metrics': {}
        }
        
        # Get unique groups
        groups = data[attribute].unique()
        
        # Calculate statistics for each group
        for group in groups:
            group_data = data[data[attribute] == group]
            
            if len(group_data) == 0:
                continue
            
            # Basic statistics
            group_stats = {
                'count': len(group_data),
                'positive_rate': group_data[target_col].mean() if target_col in group_data.columns else None,
                'prediction_mean': group_data[prediction_col].mean() if prediction_col in group_data.columns else None,
                'prediction_std': group_data[prediction_col].std() if prediction_col in group_data.columns else None
            }
            
            analysis['group_statistics'][str(group)] = group_stats
        
        # Calculate fairness metrics
        if len(analysis['group_statistics']) >= 2:
            analysis['fairness_metrics'] = self._calculate_fairness_metrics(
                analysis['group_statistics']
            )
            
            # Detect bias based on fairness metrics
            analysis['bias_detected'] = self._detect_bias_from_metrics(
                analysis['fairness_metrics']
            )
        
        return analysis
    
    def _calculate_fairness_metrics(self, group_stats: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate fairness metrics across groups.
        
        Args:
            group_stats: Statistics for each group
            
        Returns:
            Fairness metrics
        """
        metrics = {}
        
        # Get prediction means for each group
        pred_means = [stats['prediction_mean'] for stats in group_stats.values() 
                     if stats['prediction_mean'] is not None]
        
        if len(pred_means) >= 2:
            # Demographic parity difference
            metrics['demographic_parity_diff'] = max(pred_means) - min(pred_means)
            
            # Equalized odds (simplified - would need more complex calculation for full metric)
            metrics['prediction_variance'] = np.var(pred_means)
        
        return metrics
    
    def _detect_bias_from_metrics(self, fairness_metrics: Dict[str, float]) -> bool:
        """Detect bias based on fairness metrics.
        
        Args:
            fairness_metrics: Calculated fairness metrics
            
        Returns:
            True if bias is detected
        """
        # Simple thresholds for bias detection
        bias_thresholds = {
            'demographic_parity_diff': 0.1,  # 10% difference
            'prediction_variance': 0.01      # Variance threshold
        }
        
        for metric, value in fairness_metrics.items():
            threshold = bias_thresholds.get(metric, 0.1)
            if value > threshold:
                return True
        
        return False


def _json_serializer(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def generate_monitoring_report(drift_results: Dict[str, Any],
                             performance_results: Dict[str, Any],
                             bias_results: Dict[str, Any] = None,
                             output_path: str = "data/reports/monitoring_report.json") -> str:
    """Generate comprehensive monitoring report.
    
    Args:
        drift_results: Data drift detection results
        performance_results: Performance monitoring results
        bias_results: Bias monitoring results (optional)
        output_path: Path to save the report
        
    Returns:
        Path to the generated report
    """
    logger = get_logger("monitoring_report")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'drift_analysis': drift_results,
        'performance_analysis': performance_results,
        'bias_analysis': bias_results,
        'overall_status': 'healthy',
        'alerts': []
    }
    
    # Determine overall status and generate alerts
    if drift_results.get('overall_drift_detected', False):
        report['overall_status'] = 'warning'
        report['alerts'].append(f"Data drift detected in {drift_results['summary']['drifted_features']} features")
    
    if performance_results.get('degradation_detected', False):
        report['overall_status'] = 'critical'
        report['alerts'].extend(performance_results.get('alerts', []))
    
    if bias_results and bias_results.get('bias_detected', False):
        report['overall_status'] = 'warning'
        report['alerts'].append(f"Bias detected in {bias_results['summary']['biased_attributes']} attributes")
    
    # Save report
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=_json_serializer)
    
    logger.info(f"Monitoring report generated: {output_path}")
    logger.info(f"Overall status: {report['overall_status']}")
    
    return output_path


if __name__ == "__main__":
    # Example usage
    train_path = "data/processed/train.parquet"
    val_path = "data/models/lgbm/val.parquet"
    
    if file_exists(train_path) and file_exists(val_path):
        # Load data
        train_data = read_parquet(train_path)
        val_data = read_parquet(val_path)
        
        # Test drift detection
        drift_detector = DriftDetector(train_data)
        drift_results = drift_detector.detect_drift(val_data)
        
        # Test performance monitoring
        baseline_metrics = {'auroc': 0.75, 'auprc': 0.60, 'brier_score': 0.20}
        current_metrics = {'auroc': 0.73, 'auprc': 0.58, 'brier_score': 0.22}
        
        perf_monitor = PerformanceMonitor(baseline_metrics)
        perf_results = perf_monitor.add_performance_measurement(current_metrics)
        
        # Generate report
        report_path = generate_monitoring_report(drift_results, perf_results)
        print(f"Monitoring report generated: {report_path}")
    else:
        print("Required data files not found. Please run the training pipeline first.")
