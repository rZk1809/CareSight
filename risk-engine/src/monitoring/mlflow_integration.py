"""MLflow integration for experiment tracking and model registry.

This module provides comprehensive MLflow integration for the CareSight Risk Engine,
including experiment tracking, model registry, and deployment management.
"""

import os
import sys
import json
import time
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from common.io import read_parquet, file_exists
from common.logging import get_logger
from common.config import load_config

# Import MLflow with error handling
try:
    import mlflow
    import mlflow.lightgbm
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    from mlflow.models.signature import infer_signature
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow not available. Install with: pip install mlflow")


class MLflowManager:
    """Manages MLflow experiments, runs, and model registry."""
    
    def __init__(self, experiment_name: str = "caresight-risk-engine", 
                 tracking_uri: str = None):
        """Initialize MLflow manager.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (None for local)
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow not available. Install with: pip install mlflow")
        
        self.logger = get_logger("mlflow_manager")
        self.experiment_name = experiment_name
        self.client = MlflowClient()
        
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            self.logger.info(f"MLflow tracking URI set to: {tracking_uri}")
        else:
            # Use local file store (default behavior)
            tracking_dir = "mlruns"
            os.makedirs(tracking_dir, exist_ok=True)
            # On Windows, just use the default local tracking (no explicit URI)
            self.logger.info(f"MLflow tracking set to local directory: {os.path.abspath(tracking_dir)}")
        
        # Create or get experiment
        self._setup_experiment()
    
    def _setup_experiment(self):
        """Create or get the MLflow experiment."""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                self.logger.info(f"Created new experiment: {self.experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                self.logger.info(f"Using existing experiment: {self.experiment_name} (ID: {experiment_id})")
            
            mlflow.set_experiment(self.experiment_name)
            
        except Exception as e:
            self.logger.error(f"Error setting up experiment: {e}")
            raise
    
    def start_run(self, run_name: str = None, tags: Dict[str, str] = None) -> str:
        """Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Tags to add to the run
            
        Returns:
            Run ID
        """
        try:
            # Generate run name if not provided
            if run_name is None:
                run_name = f"risk_engine_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Default tags
            default_tags = {
                "model_type": "lightgbm",
                "use_case": "healthcare_risk_prediction",
                "framework": "caresight",
                "version": "1.0.0"
            }
            
            if tags:
                default_tags.update(tags)
            
            # Start run
            run = mlflow.start_run(run_name=run_name, tags=default_tags)
            run_id = run.info.run_id
            
            self.logger.info(f"Started MLflow run: {run_name} (ID: {run_id})")
            
            return run_id
            
        except Exception as e:
            self.logger.error(f"Error starting MLflow run: {e}")
            raise
    
    def log_parameters(self, params: Dict[str, Any]):
        """Log parameters to the current run.
        
        Args:
            params: Dictionary of parameters to log
        """
        try:
            for key, value in params.items():
                mlflow.log_param(key, value)
            
            self.logger.info(f"Logged {len(params)} parameters to MLflow")
            
        except Exception as e:
            self.logger.error(f"Error logging parameters: {e}")
            raise
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to the current run.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step number for the metrics
        """
        try:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                    mlflow.log_metric(key, value, step=step)
            
            self.logger.info(f"Logged {len(metrics)} metrics to MLflow")
            
        except Exception as e:
            self.logger.error(f"Error logging metrics: {e}")
            raise
    
    def log_artifacts(self, artifact_paths: List[str], artifact_path: str = None):
        """Log artifacts to the current run.
        
        Args:
            artifact_paths: List of local file paths to log
            artifact_path: Subdirectory in the run's artifact directory
        """
        try:
            for path in artifact_paths:
                if file_exists(path):
                    mlflow.log_artifact(path, artifact_path)
                else:
                    self.logger.warning(f"Artifact not found: {path}")
            
            self.logger.info(f"Logged {len(artifact_paths)} artifacts to MLflow")
            
        except Exception as e:
            self.logger.error(f"Error logging artifacts: {e}")
            raise
    
    def log_model(self, model, model_name: str, signature=None, 
                  input_example=None, registered_model_name: str = None):
        """Log a model to the current run.
        
        Args:
            model: The trained model
            model_name: Name for the model artifact
            signature: Model signature
            input_example: Example input for the model
            registered_model_name: Name for model registry
        """
        try:
            # Log the model
            if hasattr(model, 'booster'):  # LightGBM model
                mlflow.lightgbm.log_model(
                    lgb_model=model,
                    artifact_path=model_name,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name
                )
            else:  # Sklearn-compatible model
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=model_name,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name
                )
            
            self.logger.info(f"Logged model: {model_name}")
            
            if registered_model_name:
                self.logger.info(f"Registered model: {registered_model_name}")
            
        except Exception as e:
            self.logger.error(f"Error logging model: {e}")
            raise
    
    def end_run(self, status: str = "FINISHED"):
        """End the current MLflow run.
        
        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        try:
            mlflow.end_run(status=status)
            self.logger.info(f"Ended MLflow run with status: {status}")
            
        except Exception as e:
            self.logger.error(f"Error ending run: {e}")
            raise
    
    def get_best_run(self, metric_name: str, ascending: bool = False) -> Optional[Dict]:
        """Get the best run based on a metric.
        
        Args:
            metric_name: Name of the metric to optimize
            ascending: Whether to sort in ascending order
            
        Returns:
            Dictionary with run information
        """
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                return None
            
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"],
                max_results=1
            )
            
            if len(runs) == 0:
                return None
            
            best_run = runs.iloc[0]
            
            return {
                'run_id': best_run['run_id'],
                'metrics': {col.replace('metrics.', ''): best_run[col] 
                           for col in best_run.index if col.startswith('metrics.')},
                'params': {col.replace('params.', ''): best_run[col] 
                          for col in best_run.index if col.startswith('params.')},
                'tags': {col.replace('tags.', ''): best_run[col] 
                        for col in best_run.index if col.startswith('tags.')}
            }
            
        except Exception as e:
            self.logger.error(f"Error getting best run: {e}")
            return None
    
    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """Compare multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
            
        Returns:
            DataFrame with run comparison
        """
        try:
            runs_data = []
            
            for run_id in run_ids:
                run = self.client.get_run(run_id)
                
                run_data = {
                    'run_id': run_id,
                    'run_name': run.data.tags.get('mlflow.runName', 'Unknown'),
                    'start_time': datetime.fromtimestamp(run.info.start_time / 1000),
                    'status': run.info.status
                }
                
                # Add metrics
                for key, value in run.data.metrics.items():
                    run_data[f'metric_{key}'] = value
                
                # Add key parameters
                for key, value in run.data.params.items():
                    run_data[f'param_{key}'] = value
                
                runs_data.append(run_data)
            
            return pd.DataFrame(runs_data)
            
        except Exception as e:
            self.logger.error(f"Error comparing runs: {e}")
            return pd.DataFrame()
    
    def register_model(self, run_id: str, model_name: str, 
                      registered_model_name: str, stage: str = "Staging"):
        """Register a model from a run.
        
        Args:
            run_id: ID of the run containing the model
            model_name: Name of the model artifact in the run
            registered_model_name: Name for the registered model
            stage: Stage to assign to the model version
        """
        try:
            model_uri = f"runs:/{run_id}/{model_name}"
            
            # Register the model
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name
            )
            
            # Transition to specified stage
            self.client.transition_model_version_stage(
                name=registered_model_name,
                version=model_version.version,
                stage=stage
            )
            
            self.logger.info(f"Registered model {registered_model_name} "
                           f"version {model_version.version} in stage {stage}")
            
            return model_version
            
        except Exception as e:
            self.logger.error(f"Error registering model: {e}")
            raise
    
    def load_model(self, registered_model_name: str, stage: str = "Production"):
        """Load a model from the registry.
        
        Args:
            registered_model_name: Name of the registered model
            stage: Stage to load from
            
        Returns:
            Loaded model
        """
        try:
            model_uri = f"models:/{registered_model_name}/{stage}"
            model = mlflow.pyfunc.load_model(model_uri)
            
            self.logger.info(f"Loaded model {registered_model_name} from stage {stage}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def get_model_versions(self, registered_model_name: str) -> List[Dict]:
        """Get all versions of a registered model.
        
        Args:
            registered_model_name: Name of the registered model
            
        Returns:
            List of model version information
        """
        try:
            model_versions = self.client.search_model_versions(
                f"name='{registered_model_name}'"
            )
            
            versions_info = []
            for version in model_versions:
                versions_info.append({
                    'version': version.version,
                    'stage': version.current_stage,
                    'creation_timestamp': datetime.fromtimestamp(
                        int(version.creation_timestamp) / 1000
                    ),
                    'run_id': version.run_id,
                    'status': version.status
                })
            
            return versions_info
            
        except Exception as e:
            self.logger.error(f"Error getting model versions: {e}")
            return []


def track_training_run(model_path: str, calibrator_path: str, 
                      validation_path: str, metrics_path: str,
                      config_paths: List[str] = None,
                      run_name: str = None) -> str:
    """Track a complete training run with MLflow.
    
    Args:
        model_path: Path to trained model
        calibrator_path: Path to calibrator
        validation_path: Path to validation data
        metrics_path: Path to metrics JSON
        config_paths: List of configuration file paths
        run_name: Name for the MLflow run
        
    Returns:
        MLflow run ID
    """
    logger = get_logger("track_training_run")
    
    try:
        # Initialize MLflow manager
        mlflow_manager = MLflowManager()
        
        # Start run
        run_id = mlflow_manager.start_run(
            run_name=run_name,
            tags={
                "stage": "training",
                "model_type": "lightgbm_with_calibration"
            }
        )
        
        # Load and log model
        if file_exists(model_path):
            model = joblib.load(model_path)
            
            # Create model signature
            if file_exists(validation_path):
                val_data = read_parquet(validation_path)
                feature_cols = [col for col in val_data.columns 
                              if col not in ['patient', 'as_of', 'label_90d', 'prediction_proba', 'prediction']]
                
                if len(feature_cols) > 0:
                    X_sample = val_data[feature_cols].head(5)
                    y_sample = model.predict_proba(X_sample)
                    signature = infer_signature(X_sample, y_sample)
                    
                    # Log model with signature
                    mlflow_manager.log_model(
                        model=model,
                        model_name="lightgbm_model",
                        signature=signature,
                        input_example=X_sample.head(1),
                        registered_model_name="caresight-risk-model"
                    )
        
        # Log calibrator
        if file_exists(calibrator_path):
            calibrator = joblib.load(calibrator_path)
            mlflow_manager.log_model(
                model=calibrator,
                model_name="calibrator",
                registered_model_name="caresight-calibrator"
            )
        
        # Log metrics
        if file_exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
            
            if 'metrics' in metrics_data:
                mlflow_manager.log_metrics(metrics_data['metrics'])
            
            # Log confusion matrix metrics
            if 'confusion_matrices' in metrics_data:
                for threshold_name, cm_data in metrics_data['confusion_matrices'].items():
                    if 'metrics' in cm_data:
                        threshold_metrics = {
                            f"{threshold_name}_{k}": v 
                            for k, v in cm_data['metrics'].items()
                        }
                        mlflow_manager.log_metrics(threshold_metrics)
        
        # Log configuration parameters
        if config_paths:
            all_params = {}
            for config_path in config_paths:
                if file_exists(config_path):
                    config_name = Path(config_path).stem
                    config_data = load_config(config_path)
                    
                    # Flatten config for MLflow
                    flattened = _flatten_dict(config_data, prefix=f"{config_name}_")
                    all_params.update(flattened)
            
            if all_params:
                mlflow_manager.log_parameters(all_params)
        
        # Log artifacts
        artifacts = [model_path, calibrator_path, validation_path, metrics_path]
        if config_paths:
            artifacts.extend(config_paths)
        
        existing_artifacts = [path for path in artifacts if file_exists(path)]
        if existing_artifacts:
            mlflow_manager.log_artifacts(existing_artifacts)
        
        # End run
        mlflow_manager.end_run("FINISHED")
        
        logger.info(f"Training run tracked successfully: {run_id}")
        
        return run_id
        
    except Exception as e:
        logger.error(f"Error tracking training run: {e}")
        if 'mlflow_manager' in locals():
            mlflow_manager.end_run("FAILED")
        raise


def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested dictionary for MLflow parameters.
    
    Args:
        d: Dictionary to flatten
        prefix: Prefix for keys
        
    Returns:
        Flattened dictionary
    """
    flattened = {}
    
    for key, value in d.items():
        new_key = f"{prefix}{key}" if prefix else key
        
        if isinstance(value, dict):
            flattened.update(_flatten_dict(value, f"{new_key}_"))
        elif isinstance(value, (list, tuple)):
            flattened[new_key] = str(value)
        else:
            flattened[new_key] = value
    
    return flattened


def compare_model_performance(experiment_name: str = "caresight-risk-engine") -> pd.DataFrame:
    """Compare performance of all models in the experiment.
    
    Args:
        experiment_name: Name of the MLflow experiment
        
    Returns:
        DataFrame with model comparison
    """
    logger = get_logger("compare_models")
    
    try:
        mlflow_manager = MLflowManager(experiment_name)
        
        # Get all runs
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning(f"Experiment {experiment_name} not found")
            return pd.DataFrame()
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        
        if len(runs) == 0:
            logger.info("No runs found in experiment")
            return pd.DataFrame()
        
        # Select relevant columns
        comparison_cols = ['run_id', 'start_time', 'status']
        
        # Add metric columns
        metric_cols = [col for col in runs.columns if col.startswith('metrics.')]
        comparison_cols.extend(metric_cols)
        
        # Add key parameter columns
        param_cols = [col for col in runs.columns if col.startswith('params.') and 
                     any(key in col for key in ['learning_rate', 'n_estimators', 'max_depth'])]
        comparison_cols.extend(param_cols)
        
        # Filter and clean column names
        available_cols = [col for col in comparison_cols if col in runs.columns]
        comparison_df = runs[available_cols].copy()
        
        # Rename columns for readability
        comparison_df.columns = [col.replace('metrics.', '').replace('params.', 'param_') 
                               for col in comparison_df.columns]
        
        logger.info(f"Compared {len(comparison_df)} model runs")
        
        return comparison_df
        
    except Exception as e:
        logger.error(f"Error comparing model performance: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    model_path = "data/models/lgbm/lgbm.pkl"
    calibrator_path = "data/models/lgbm/calibrator_isotonic.pkl"
    validation_path = "data/models/lgbm/val.parquet"
    metrics_path = "data/reports/metrics.json"
    
    config_paths = [
        "configs/data.yaml",
        "configs/model_lightgbm.yaml",
        "configs/thresholds.yaml"
    ]
    
    if all(file_exists(path) for path in [model_path, calibrator_path, validation_path, metrics_path]):
        run_id = track_training_run(
            model_path=model_path,
            calibrator_path=calibrator_path,
            validation_path=validation_path,
            metrics_path=metrics_path,
            config_paths=config_paths,
            run_name="test_tracking_run"
        )
        print(f"Tracked training run: {run_id}")
        
        # Compare models
        comparison = compare_model_performance()
        print(f"Model comparison:\n{comparison}")
    else:
        print("Required files not found. Please run the training pipeline first.")
