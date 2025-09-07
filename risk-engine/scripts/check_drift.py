"""Script to check for data drift in production data."""

import sys
import os
import json
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from monitoring.drift_detection import DriftDetector
from common.io import file_exists, read_parquet
from common.logging import get_logger


def check_drift():
    """Check for data drift and return exit code based on results."""
    logger = get_logger("check_drift")
    
    try:
        # Paths
        reference_data_path = "data/processed/train.parquet"
        current_data_path = "data/processed/current.parquet"  # This would be production data
        
        # For demo, use validation data as current data
        if not file_exists(current_data_path):
            current_data_path = "data/models/lgbm/val.parquet"
        
        if not file_exists(reference_data_path) or not file_exists(current_data_path):
            logger.error("Required data files not found")
            return 1
        
        # Load data
        reference_data = read_parquet(reference_data_path)
        current_data = read_parquet(current_data_path)
        
        # Initialize drift detector
        drift_detector = DriftDetector(reference_data)
        
        # Detect drift
        drift_results = drift_detector.detect_drift(current_data)
        
        # Save results
        output_path = "data/reports/drift_report.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(drift_results, f, indent=2, default=str)
        
        # Log results
        logger.info(f"Drift detection completed")
        logger.info(f"Overall drift detected: {drift_results['overall_drift_detected']}")
        logger.info(f"Drift rate: {drift_results['summary']['drift_rate']:.2%}")
        
        # Return appropriate exit code
        if drift_results['overall_drift_detected']:
            logger.warning("Significant data drift detected - retraining recommended")
            return 2  # Warning level
        else:
            logger.info("No significant drift detected")
            return 0  # Success
            
    except Exception as e:
        logger.error(f"Drift check failed: {e}")
        return 1  # Error


if __name__ == "__main__":
    exit_code = check_drift()
    sys.exit(exit_code)
