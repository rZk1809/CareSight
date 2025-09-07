"""Automated model retraining script."""

import sys
import os
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from common.io import file_exists
from common.logging import get_logger


def should_retrain():
    """Determine if model should be retrained based on drift and performance."""
    logger = get_logger("should_retrain")
    
    # Check drift report
    drift_report_path = "data/reports/drift_report.json"
    if file_exists(drift_report_path):
        with open(drift_report_path, 'r') as f:
            drift_results = json.load(f)
        
        if drift_results.get('overall_drift_detected', False):
            logger.info("Data drift detected - retraining recommended")
            return True
    
    # Check performance degradation
    monitoring_report_path = "data/reports/monitoring_report.json"
    if file_exists(monitoring_report_path):
        with open(monitoring_report_path, 'r') as f:
            monitoring_results = json.load(f)
        
        if monitoring_results.get('overall_status') == 'critical':
            logger.info("Performance degradation detected - retraining required")
            return True
    
    # Check time since last training
    model_path = "data/models/lgbm/lgbm.pkl"
    if file_exists(model_path):
        model_age_days = (datetime.now().timestamp() - os.path.getmtime(model_path)) / (24 * 3600)
        if model_age_days > 30:  # Retrain if model is older than 30 days
            logger.info(f"Model is {model_age_days:.1f} days old - retraining recommended")
            return True
    
    logger.info("No retraining needed")
    return False


def run_retraining():
    """Run the complete retraining pipeline."""
    logger = get_logger("run_retraining")
    
    try:
        logger.info("Starting automated retraining...")
        
        # Run DVC pipeline
        logger.info("Running DVC pipeline...")
        result = subprocess.run(["dvc", "repro"], capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"DVC pipeline failed: {result.stderr}")
            return False
        
        logger.info("DVC pipeline completed successfully")
        
        # Verify new model exists
        model_path = "data/models/lgbm/lgbm.pkl"
        if not file_exists(model_path):
            logger.error("New model not found after retraining")
            return False
        
        logger.info("Model retraining completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        return False


def update_model_metadata():
    """Update model metadata after retraining."""
    logger = get_logger("update_metadata")
    
    try:
        metadata = {
            "retrained_at": datetime.now().isoformat(),
            "retrained_by": "automated_pipeline",
            "trigger": "scheduled_retraining",
            "version": f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        metadata_path = "data/models/lgbm/metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model metadata updated: {metadata_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update metadata: {e}")
        return False


def main():
    """Main retraining workflow."""
    logger = get_logger("automated_retraining")
    
    try:
        # Check if retraining is needed
        if not should_retrain():
            logger.info("Retraining not needed - exiting")
            return 0
        
        # Run retraining
        if not run_retraining():
            logger.error("Retraining failed")
            return 1
        
        # Update metadata
        if not update_model_metadata():
            logger.warning("Failed to update metadata, but retraining succeeded")
        
        logger.info("Automated retraining completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Automated retraining failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
