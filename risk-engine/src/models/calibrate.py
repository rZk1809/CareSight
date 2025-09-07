"""Calibrate model probabilities using isotonic regression."""

import argparse
import joblib
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from common.io import read_parquet
from common.logging import get_logger, log_execution_time


def load_model_and_validation_data(model_path: str, validation_path: str, logger) -> tuple:
    """Load trained model and validation data.
    
    Args:
        model_path: Path to trained model pickle file
        validation_path: Path to validation data parquet file
        logger: Logger instance
        
    Returns:
        Tuple of (model, validation_df)
    """
    logger.info("Loading model and validation data...")
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    # Load validation data
    logger.info(f"Loading validation data from {validation_path}")
    validation_df = read_parquet(validation_path)
    
    logger.info(f"Validation data shape: {validation_df.shape}")
    
    return model, validation_df


def extract_probabilities_and_labels(validation_df: pd.DataFrame, logger) -> tuple:
    """Extract probabilities and true labels from validation data.
    
    Args:
        validation_df: Validation DataFrame with predictions
        logger: Logger instance
        
    Returns:
        Tuple of (probabilities, true_labels)
    """
    logger.info("Extracting probabilities and labels...")
    
    # Check required columns
    required_cols = ["label_90d", "prediction_proba"]
    missing_cols = [col for col in required_cols if col not in validation_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Extract data
    true_labels = validation_df["label_90d"].values
    probabilities = validation_df["prediction_proba"].values
    
    logger.info(f"Number of samples: {len(true_labels)}")
    logger.info(f"Probability range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")
    
    # Log label distribution
    unique_labels, counts = np.unique(true_labels, return_counts=True)
    label_dist = dict(zip(unique_labels, counts))
    logger.info(f"Label distribution: {label_dist}")
    
    return probabilities, true_labels


def fit_isotonic_calibrator(probabilities: np.ndarray, true_labels: np.ndarray, logger) -> IsotonicRegression:
    """Fit isotonic regression calibrator.
    
    Args:
        probabilities: Raw model probabilities
        true_labels: True binary labels
        logger: Logger instance
        
    Returns:
        Fitted isotonic regression calibrator
    """
    logger.info("Fitting isotonic regression calibrator...")
    
    # Initialize isotonic regression
    calibrator = IsotonicRegression(out_of_bounds='clip')
    
    # Fit calibrator
    calibrator.fit(probabilities, true_labels)
    
    # Get calibrated probabilities for analysis
    calibrated_probs = calibrator.predict(probabilities)
    
    logger.info(f"Calibrated probability range: [{calibrated_probs.min():.4f}, {calibrated_probs.max():.4f}]")
    
    # Log some statistics about the calibration
    prob_bins = np.linspace(0, 1, 11)
    for i in range(len(prob_bins) - 1):
        bin_mask = (probabilities >= prob_bins[i]) & (probabilities < prob_bins[i+1])
        if bin_mask.sum() > 0:
            bin_true_rate = true_labels[bin_mask].mean()
            bin_mean_prob = probabilities[bin_mask].mean()
            bin_mean_calib = calibrated_probs[bin_mask].mean()
            logger.info(f"Bin [{prob_bins[i]:.1f}, {prob_bins[i+1]:.1f}): "
                       f"n={bin_mask.sum()}, true_rate={bin_true_rate:.3f}, "
                       f"mean_prob={bin_mean_prob:.3f}, mean_calib={bin_mean_calib:.3f}")
    
    return calibrator


def save_calibrator(calibrator: IsotonicRegression, output_path: str, logger) -> None:
    """Save calibrator to disk.
    
    Args:
        calibrator: Fitted calibrator
        output_path: Path to save calibrator
        logger: Logger instance
    """
    logger.info(f"Saving calibrator to {output_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save calibrator
    joblib.dump(calibrator, output_path)
    
    logger.info("Calibrator saved successfully")


def calibrate_model(
    model_path: str,
    validation_path: str,
    output_path: str
) -> None:
    """Calibrate model probabilities using isotonic regression.
    
    Args:
        model_path: Path to trained model pickle file
        validation_path: Path to validation data parquet file
        output_path: Path to save calibrator pickle file
    """
    start_time = time.time()
    logger = get_logger("calibrate")
    
    logger.info("Starting model calibration process...")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Validation path: {validation_path}")
    logger.info(f"Output path: {output_path}")
    
    # Load model and validation data
    model, validation_df = load_model_and_validation_data(model_path, validation_path, logger)
    
    # Extract probabilities and labels
    probabilities, true_labels = extract_probabilities_and_labels(validation_df, logger)
    
    # Fit calibrator
    calibrator = fit_isotonic_calibrator(probabilities, true_labels, logger)
    
    # Save calibrator
    save_calibrator(calibrator, output_path, logger)
    
    end_time = time.time()
    log_execution_time(logger, start_time, end_time, "Model calibration")
    
    logger.info("Model calibration completed successfully!")


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description="Calibrate model probabilities")
    parser.add_argument("--model", required=True, help="Path to trained model pickle file")
    parser.add_argument("--validation", required=True, help="Path to validation data parquet file")
    parser.add_argument("--output", required=True, help="Output path for calibrator pickle file")
    
    args = parser.parse_args()
    
    calibrate_model(
        model_path=args.model,
        validation_path=args.validation,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
