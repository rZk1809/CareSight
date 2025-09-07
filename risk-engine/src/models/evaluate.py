"""Evaluate model performance with metrics and confusion matrices."""

import argparse
import joblib
import json
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report
)

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from common.io import read_parquet, ensure_dir
from common.logging import get_logger, log_execution_time


def load_model_artifacts(model_path: str, calibrator_path: str, validation_path: str, logger) -> tuple:
    """Load model, calibrator, and validation data.
    
    Args:
        model_path: Path to trained model
        calibrator_path: Path to calibrator
        validation_path: Path to validation data
        logger: Logger instance
        
    Returns:
        Tuple of (model, calibrator, validation_df)
    """
    logger.info("Loading model artifacts...")
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    # Load calibrator
    logger.info(f"Loading calibrator from {calibrator_path}")
    calibrator = joblib.load(calibrator_path)
    
    # Load validation data
    logger.info(f"Loading validation data from {validation_path}")
    validation_df = read_parquet(validation_path)
    
    logger.info(f"Validation data shape: {validation_df.shape}")
    
    return model, calibrator, validation_df


def compute_predictions(model, calibrator, validation_df: pd.DataFrame, logger) -> tuple:
    """Compute raw and calibrated predictions.
    
    Args:
        model: Trained model
        calibrator: Fitted calibrator
        validation_df: Validation DataFrame
        logger: Logger instance
        
    Returns:
        Tuple of (y_true, y_prob_raw, y_prob_calibrated)
    """
    logger.info("Computing predictions...")
    
    # Extract true labels
    y_true = validation_df["label_90d"].values
    
    # Get feature columns (exclude ID and target columns)
    feature_cols = [col for col in validation_df.columns 
                   if col not in ["patient", "as_of", "label_90d", "prediction_proba", "prediction"]]
    X = validation_df[feature_cols]
    
    # Compute raw probabilities
    y_prob_raw = model.predict_proba(X)[:, 1]
    
    # Compute calibrated probabilities
    y_prob_calibrated = calibrator.predict(y_prob_raw)
    
    logger.info(f"Raw probability range: [{y_prob_raw.min():.4f}, {y_prob_raw.max():.4f}]")
    logger.info(f"Calibrated probability range: [{y_prob_calibrated.min():.4f}, {y_prob_calibrated.max():.4f}]")
    
    return y_true, y_prob_raw, y_prob_calibrated


def compute_metrics(y_true: np.ndarray, y_prob_raw: np.ndarray, y_prob_calibrated: np.ndarray, logger) -> dict:
    """Compute evaluation metrics.
    
    Args:
        y_true: True binary labels
        y_prob_raw: Raw model probabilities
        y_prob_calibrated: Calibrated probabilities
        logger: Logger instance
        
    Returns:
        Dictionary with computed metrics
    """
    logger.info("Computing evaluation metrics...")
    
    # ROC AUC
    auroc_raw = roc_auc_score(y_true, y_prob_raw)
    auroc_calibrated = roc_auc_score(y_true, y_prob_calibrated)
    
    # Average Precision (AUPRC)
    auprc_raw = average_precision_score(y_true, y_prob_raw)
    auprc_calibrated = average_precision_score(y_true, y_prob_calibrated)
    
    # Brier Score (lower is better)
    brier_raw = brier_score_loss(y_true, y_prob_raw)
    brier_calibrated = brier_score_loss(y_true, y_prob_calibrated)
    
    metrics = {
        "auroc_raw": float(auroc_raw),
        "auroc_calibrated": float(auroc_calibrated),
        "auprc_raw": float(auprc_raw),
        "auprc_calibrated": float(auprc_calibrated),
        "brier_raw": float(brier_raw),
        "brier_calibrated": float(brier_calibrated)
    }
    
    logger.info("Metrics computed:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    return metrics


def compute_confusion_matrices(
    y_true: np.ndarray, 
    y_prob_calibrated: np.ndarray, 
    thresholds: dict, 
    logger
) -> dict:
    """Compute confusion matrices at different thresholds.
    
    Args:
        y_true: True binary labels
        y_prob_calibrated: Calibrated probabilities
        thresholds: Dictionary of threshold names and values
        logger: Logger instance
        
    Returns:
        Dictionary with confusion matrices and derived metrics
    """
    logger.info("Computing confusion matrices...")
    
    confusion_results = {}
    
    for threshold_name, threshold_value in thresholds.items():
        logger.info(f"Computing confusion matrix for {threshold_name} threshold: {threshold_value}")
        
        # Apply threshold
        y_pred = (y_prob_calibrated >= threshold_value).astype(int)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        # Compute derived metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        
        confusion_results[threshold_name] = {
            "threshold": float(threshold_value),
            "confusion_matrix": {
                "tn": int(tn), "fp": int(fp),
                "fn": int(fn), "tp": int(tp)
            },
            "metrics": {
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "ppv": float(ppv),
                "npv": float(npv),
                "f1": float(f1)
            }
        }
        
        logger.info(f"  Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        logger.info(f"  Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}")
        logger.info(f"  PPV: {ppv:.3f}, NPV: {npv:.3f}, F1: {f1:.3f}")
    
    return confusion_results


def save_evaluation_results(metrics: dict, confusion_results: dict, output_path: str, logger) -> None:
    """Save evaluation results to JSON file.
    
    Args:
        metrics: Dictionary with evaluation metrics
        confusion_results: Dictionary with confusion matrix results
        output_path: Path to save results JSON file
        logger: Logger instance
    """
    logger.info(f"Saving evaluation results to {output_path}")
    
    # Ensure output directory exists
    ensure_dir(os.path.dirname(output_path))
    
    # Combine results
    results = {
        "metrics": metrics,
        "confusion_matrices": confusion_results
    }
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Evaluation results saved successfully")


def evaluate_model(
    model_path: str,
    calibrator_path: str,
    validation_path: str,
    high_sensitivity_threshold: float,
    high_ppv_threshold: float,
    output_path: str
) -> None:
    """Evaluate model performance.
    
    Args:
        model_path: Path to trained model
        calibrator_path: Path to calibrator
        validation_path: Path to validation data
        high_sensitivity_threshold: Threshold for high sensitivity
        high_ppv_threshold: Threshold for high PPV
        output_path: Path to save evaluation results
    """
    start_time = time.time()
    logger = get_logger("evaluate")
    
    logger.info("Starting model evaluation process...")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Calibrator path: {calibrator_path}")
    logger.info(f"Validation path: {validation_path}")
    logger.info(f"High sensitivity threshold: {high_sensitivity_threshold}")
    logger.info(f"High PPV threshold: {high_ppv_threshold}")
    logger.info(f"Output path: {output_path}")
    
    # Load artifacts
    model, calibrator, validation_df = load_model_artifacts(
        model_path, calibrator_path, validation_path, logger
    )
    
    # Compute predictions
    y_true, y_prob_raw, y_prob_calibrated = compute_predictions(
        model, calibrator, validation_df, logger
    )
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_prob_raw, y_prob_calibrated, logger)
    
    # Compute confusion matrices
    thresholds = {
        "high_sensitivity": high_sensitivity_threshold,
        "high_ppv": high_ppv_threshold
    }
    confusion_results = compute_confusion_matrices(y_true, y_prob_calibrated, thresholds, logger)
    
    # Save results
    save_evaluation_results(metrics, confusion_results, output_path, logger)
    
    end_time = time.time()
    log_execution_time(logger, start_time, end_time, "Model evaluation")
    
    logger.info("Model evaluation completed successfully!")


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description="Evaluate model performance")
    parser.add_argument("--model", required=True, help="Path to trained model pickle file")
    parser.add_argument("--calibrator", required=True, help="Path to calibrator pickle file")
    parser.add_argument("--validation", required=True, help="Path to validation data parquet file")
    parser.add_argument("--high-sensitivity-threshold", type=float, required=True, 
                       help="Threshold for high sensitivity")
    parser.add_argument("--high-ppv-threshold", type=float, required=True,
                       help="Threshold for high PPV")
    parser.add_argument("--output", required=True, help="Output path for evaluation results JSON")
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        calibrator_path=args.calibrator,
        validation_path=args.validation,
        high_sensitivity_threshold=args.high_sensitivity_threshold,
        high_ppv_threshold=args.high_ppv_threshold,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
