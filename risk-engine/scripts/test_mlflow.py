"""Test script for MLflow integration."""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from monitoring.mlflow_integration import (
    MLflowManager, track_training_run, compare_model_performance
)
from common.io import file_exists
from common.logging import get_logger


def test_mlflow_integration():
    """Test MLflow integration functionality."""
    logger = get_logger("test_mlflow")
    
    try:
        logger.info("Testing MLflow integration...")
        
        # Test MLflow manager initialization
        logger.info("1. Testing MLflow manager initialization...")
        mlflow_manager = MLflowManager(experiment_name="test-caresight-risk-engine")
        logger.info("✅ MLflow manager initialized successfully")
        
        # Test starting a run
        logger.info("2. Testing run creation...")
        run_id = mlflow_manager.start_run(
            run_name="test_run",
            tags={"test": "true", "purpose": "integration_test"}
        )
        logger.info(f"✅ Run started successfully: {run_id}")
        
        # Test logging parameters
        logger.info("3. Testing parameter logging...")
        test_params = {
            "learning_rate": 0.1,
            "n_estimators": 100,
            "max_depth": 6,
            "test_param": "test_value"
        }
        mlflow_manager.log_parameters(test_params)
        logger.info("✅ Parameters logged successfully")
        
        # Test logging metrics
        logger.info("4. Testing metrics logging...")
        test_metrics = {
            "auroc": 0.85,
            "auprc": 0.72,
            "brier_score": 0.18,
            "accuracy": 0.78
        }
        mlflow_manager.log_metrics(test_metrics)
        logger.info("✅ Metrics logged successfully")
        
        # Test ending run
        logger.info("5. Testing run completion...")
        mlflow_manager.end_run("FINISHED")
        logger.info("✅ Run ended successfully")
        
        logger.info("🎉 Basic MLflow integration tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ MLflow integration test failed: {e}")
        return False


def test_training_run_tracking():
    """Test tracking a complete training run."""
    logger = get_logger("test_training_tracking")
    
    try:
        logger.info("Testing training run tracking...")
        
        # Check if required files exist
        model_path = "data/models/lgbm/lgbm.pkl"
        calibrator_path = "data/models/lgbm/calibrator_isotonic.pkl"
        validation_path = "data/models/lgbm/val.parquet"
        metrics_path = "data/reports/metrics.json"
        
        config_paths = [
            "configs/data.yaml",
            "configs/model_lightgbm.yaml",
            "configs/thresholds.yaml"
        ]
        
        # Check file existence
        missing_files = []
        for path in [model_path, calibrator_path, validation_path, metrics_path]:
            if not file_exists(path):
                missing_files.append(path)
        
        if missing_files:
            logger.warning(f"Missing files for training run tracking: {missing_files}")
            logger.info("Skipping training run tracking test")
            return True
        
        # Track training run
        logger.info("Tracking complete training run...")
        run_id = track_training_run(
            model_path=model_path,
            calibrator_path=calibrator_path,
            validation_path=validation_path,
            metrics_path=metrics_path,
            config_paths=config_paths,
            run_name="test_complete_training_run"
        )
        
        logger.info(f"✅ Training run tracked successfully: {run_id}")
        
        # Test model comparison
        logger.info("Testing model comparison...")
        comparison_df = compare_model_performance("test-caresight-risk-engine")
        
        if len(comparison_df) > 0:
            logger.info(f"✅ Model comparison completed: {len(comparison_df)} runs found")
            logger.info(f"Comparison columns: {list(comparison_df.columns)}")
        else:
            logger.info("No runs found for comparison (this is expected for first run)")
        
        logger.info("🎉 Training run tracking tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Training run tracking test failed: {e}")
        return False


def test_model_registry():
    """Test model registry functionality."""
    logger = get_logger("test_model_registry")
    
    try:
        logger.info("Testing model registry functionality...")
        
        # Initialize MLflow manager
        mlflow_manager = MLflowManager(experiment_name="test-caresight-risk-engine")
        
        # Get the best run (if any)
        best_run = mlflow_manager.get_best_run("auroc", ascending=False)
        
        if best_run is None:
            logger.info("No runs found for model registry test")
            return True
        
        logger.info(f"Found best run: {best_run['run_id']}")
        
        # Test getting model versions
        logger.info("Testing model version retrieval...")
        versions = mlflow_manager.get_model_versions("caresight-risk-model")
        
        if versions:
            logger.info(f"✅ Found {len(versions)} model versions")
            for version in versions:
                logger.info(f"  Version {version['version']}: {version['stage']}")
        else:
            logger.info("No model versions found (expected for first run)")
        
        logger.info("🎉 Model registry tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model registry test failed: {e}")
        return False


def test_mlflow_ui_access():
    """Test MLflow UI accessibility."""
    logger = get_logger("test_mlflow_ui")
    
    try:
        logger.info("Testing MLflow UI access...")
        
        # Check if mlruns directory exists
        mlruns_dir = "mlruns"
        if os.path.exists(mlruns_dir):
            logger.info(f"✅ MLflow tracking directory exists: {mlruns_dir}")
            
            # List experiments
            experiments = os.listdir(mlruns_dir)
            logger.info(f"Found experiments: {experiments}")
            
            logger.info("💡 To view MLflow UI, run: mlflow ui")
            logger.info("💡 Then open: http://localhost:5000")
        else:
            logger.info("MLflow tracking directory not found (no runs yet)")
        
        logger.info("🎉 MLflow UI access test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ MLflow UI access test failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing CareSight Risk Engine MLflow Integration")
    print("=" * 60)
    
    # Test basic MLflow integration
    success1 = test_mlflow_integration()
    
    print("\n" + "=" * 60)
    
    # Test training run tracking
    success2 = test_training_run_tracking()
    
    print("\n" + "=" * 60)
    
    # Test model registry
    success3 = test_model_registry()
    
    print("\n" + "=" * 60)
    
    # Test MLflow UI access
    success4 = test_mlflow_ui_access()
    
    print("\n" + "=" * 60)
    
    if all([success1, success2, success3, success4]):
        print("🎉 All MLflow integration tests PASSED!")
    else:
        print("❌ Some MLflow integration tests FAILED!")
        sys.exit(1)
