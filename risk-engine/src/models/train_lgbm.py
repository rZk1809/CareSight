"""Train LightGBM model for risk prediction."""

import argparse
import joblib
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from common.io import read_parquet, write_parquet, ensure_dir
from common.logging import get_logger, log_dataframe_info, log_execution_time


def prepare_training_data(
    dataset_df: pd.DataFrame,
    target_col: str,
    id_cols: list,
    logger
) -> tuple:
    """Prepare features and target for training.
    
    Args:
        dataset_df: Training dataset
        target_col: Name of target column
        id_cols: List of ID columns to exclude from features
        logger: Logger instance
        
    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    logger.info("Preparing training data...")
    
    # Check target column exists
    if target_col not in dataset_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    # Extract target
    y = dataset_df[target_col].astype(int)
    
    # Remove target and ID columns from features
    feature_cols = [col for col in dataset_df.columns if col not in [target_col] + id_cols]
    X = dataset_df[feature_cols].copy()
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    logger.info(f"Feature columns: {list(X.columns)}")
    
    # Log target distribution
    target_counts = y.value_counts().sort_index()
    logger.info(f"Target distribution: {target_counts.to_dict()}")
    
    # Check for missing values in features
    missing_counts = X.isnull().sum()
    features_with_missing = missing_counts[missing_counts > 0]
    if len(features_with_missing) > 0:
        logger.info(f"Features with missing values: {features_with_missing.to_dict()}")
    
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float, stratify: bool, random_state: int, logger) -> tuple:
    """Split data into training and validation sets.
    
    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Fraction for validation set
        stratify: Whether to stratify split
        random_state: Random seed
        logger: Logger instance
        
    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
    """
    logger.info(f"Splitting data with test_size={test_size}, stratify={stratify}")
    
    stratify_y = y if stratify else None
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        stratify=stratify_y,
        random_state=random_state
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Validation set: {X_val.shape[0]} samples")
    
    # Log target distribution in splits
    train_dist = y_train.value_counts().sort_index()
    val_dist = y_val.value_counts().sort_index()
    logger.info(f"Training target distribution: {train_dist.to_dict()}")
    logger.info(f"Validation target distribution: {val_dist.to_dict()}")
    
    return X_train, X_val, y_train, y_val


def train_lightgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_params: dict,
    logger
) -> LGBMClassifier:
    """Train LightGBM model.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        model_params: Model hyperparameters
        logger: Logger instance
        
    Returns:
        Trained LightGBM model
    """
    logger.info("Training LightGBM model...")
    logger.info(f"Model parameters: {model_params}")
    
    # Initialize model
    model = LGBMClassifier(**model_params)
    
    # Train model
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[
            # LightGBM will use early stopping if early_stopping_rounds is in params
        ]
    )
    
    # Log feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("Top 10 most important features:")
    for _, row in feature_importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    return model


def save_model_artifacts(
    model: LGBMClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    output_dir: str,
    logger
) -> None:
    """Save model and validation data.
    
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation target
        output_dir: Output directory
        logger: Logger instance
    """
    logger.info(f"Saving model artifacts to {output_dir}")
    
    # Ensure output directory exists
    ensure_dir(output_dir)
    
    # Save model
    model_path = os.path.join(output_dir, "lgbm.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Save validation data with predictions
    val_data = X_val.copy()
    val_data["label_90d"] = y_val
    val_data["prediction_proba"] = model.predict_proba(X_val)[:, 1]
    val_data["prediction"] = model.predict(X_val)
    
    val_path = os.path.join(output_dir, "val.parquet")
    write_parquet(val_data, val_path)
    logger.info(f"Saved validation data to {val_path}")


def train_lgbm(
    dataset_path: str,
    target: str,
    id_cols: str,
    validation_split: float,
    stratify: bool,
    model_params: dict,
    output_dir: str
) -> None:
    """Train LightGBM model pipeline.
    
    Args:
        dataset_path: Path to training dataset
        target: Target column name
        id_cols: Comma-separated ID columns
        validation_split: Validation set fraction
        stratify: Whether to stratify split
        model_params: Model hyperparameters
        output_dir: Output directory for model artifacts
    """
    start_time = time.time()
    logger = get_logger("train_lgbm")
    
    logger.info("Starting LightGBM training pipeline...")
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Target: {target}")
    logger.info(f"ID columns: {id_cols}")
    logger.info(f"Validation split: {validation_split}")
    logger.info(f"Output directory: {output_dir}")
    
    # Parse ID columns
    id_col_list = [col.strip() for col in id_cols.split(",")]
    
    # Load dataset
    logger.info("Loading training dataset...")
    dataset = read_parquet(dataset_path)
    log_dataframe_info(logger, dataset, "Training dataset")
    
    # Prepare data
    X, y = prepare_training_data(dataset, target, id_col_list, logger)
    
    # Split data
    X_train, X_val, y_train, y_val = split_data(
        X, y, validation_split, stratify, model_params.get('random_state', 42), logger
    )
    
    # Train model
    model = train_lightgbm_model(X_train, y_train, X_val, y_val, model_params, logger)
    
    # Save artifacts
    save_model_artifacts(model, X_val, y_val, output_dir, logger)
    
    end_time = time.time()
    log_execution_time(logger, start_time, end_time, "LightGBM training")
    
    logger.info("LightGBM training completed successfully!")


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description="Train LightGBM model")
    parser.add_argument("--dataset", required=True, help="Path to training dataset parquet file")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--id-cols", required=True, help="Comma-separated ID columns")
    parser.add_argument("--validation-split", type=float, default=0.25, help="Validation set fraction")
    parser.add_argument("--stratify", action="store_true", help="Stratify the split")
    parser.add_argument("--output-dir", required=True, help="Output directory for model artifacts")
    
    # Model parameters
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--min-child-samples", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--class-weight", default="balanced")
    
    args = parser.parse_args()
    
    # Build model parameters dict
    model_params = {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "num_leaves": args.num_leaves,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_lambda": args.reg_lambda,
        "min_child_samples": args.min_child_samples,
        "random_state": args.random_state,
        "class_weight": args.class_weight,
    }
    
    train_lgbm(
        dataset_path=args.dataset,
        target=args.target,
        id_cols=args.id_cols,
        validation_split=args.validation_split,
        stratify=args.stratify,
        model_params=model_params,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
