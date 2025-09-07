"""Merge features and labels into a single training table."""

import argparse
import time
import pandas as pd
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from common.io import read_parquet, write_parquet
from common.logging import get_logger, log_dataframe_info, log_execution_time


def validate_merge_keys(features_df: pd.DataFrame, labels_df: pd.DataFrame, logger) -> None:
    """Validate that merge keys are consistent between features and labels.
    
    Args:
        features_df: Features DataFrame
        labels_df: Labels DataFrame
        logger: Logger instance
    """
    logger.info("Validating merge keys...")
    
    # Check for required columns
    required_cols = ["patient", "as_of"]
    
    for col in required_cols:
        if col not in features_df.columns:
            raise ValueError(f"Missing required column '{col}' in features")
        if col not in labels_df.columns:
            raise ValueError(f"Missing required column '{col}' in labels")
    
    # Check for duplicates
    features_dupes = features_df.duplicated(subset=required_cols).sum()
    labels_dupes = labels_df.duplicated(subset=required_cols).sum()
    
    if features_dupes > 0:
        logger.warning(f"Found {features_dupes} duplicate (patient, as_of) pairs in features")
    if labels_dupes > 0:
        logger.warning(f"Found {labels_dupes} duplicate (patient, as_of) pairs in labels")
    
    # Check overlap
    features_keys = set(features_df[required_cols].apply(tuple, axis=1))
    labels_keys = set(labels_df[required_cols].apply(tuple, axis=1))
    
    overlap = features_keys.intersection(labels_keys)
    features_only = features_keys - labels_keys
    labels_only = labels_keys - features_keys
    
    logger.info(f"Keys in both features and labels: {len(overlap)}")
    logger.info(f"Keys only in features: {len(features_only)}")
    logger.info(f"Keys only in labels: {len(labels_only)}")
    
    if len(features_only) > 0:
        logger.warning("Some patients in features are missing from labels")
    if len(labels_only) > 0:
        logger.warning("Some patients in labels are missing from features")


def merge_features_labels(features_df: pd.DataFrame, labels_df: pd.DataFrame, logger) -> pd.DataFrame:
    """Merge features and labels DataFrames.
    
    Args:
        features_df: Features DataFrame
        labels_df: Labels DataFrame
        logger: Logger instance
        
    Returns:
        Merged training DataFrame
    """
    logger.info("Merging features and labels...")
    
    # Perform inner join to keep only patients with both features and labels
    merged_df = features_df.merge(
        labels_df,
        on=["patient", "as_of"],
        how="inner"
    )
    
    logger.info(f"Merged dataset shape: {merged_df.shape}")
    logger.info(f"Features shape: {features_df.shape}")
    logger.info(f"Labels shape: {labels_df.shape}")
    
    # Check for any missing values in the target
    if "label_90d" in merged_df.columns:
        missing_labels = merged_df["label_90d"].isna().sum()
        if missing_labels > 0:
            logger.warning(f"Found {missing_labels} missing labels in merged dataset")
            # Remove rows with missing labels
            merged_df = merged_df.dropna(subset=["label_90d"])
            logger.info(f"After removing missing labels: {merged_df.shape}")
    
    return merged_df


def analyze_training_table(training_df: pd.DataFrame, logger) -> None:
    """Analyze the final training table.
    
    Args:
        training_df: Training DataFrame
        logger: Logger instance
    """
    logger.info("Analyzing training table...")
    
    # Basic info
    log_dataframe_info(logger, training_df, "Training table")
    
    # Target distribution
    if "label_90d" in training_df.columns:
        label_counts = training_df["label_90d"].value_counts().sort_index()
        logger.info(f"Label distribution: {label_counts.to_dict()}")
        
        positive_rate = training_df["label_90d"].mean()
        logger.info(f"Positive rate: {positive_rate:.3f} ({positive_rate*100:.1f}%)")
    
    # Feature completeness
    logger.info("Feature completeness:")
    feature_cols = [col for col in training_df.columns if col not in ["patient", "as_of", "label_90d"]]
    
    for col in feature_cols:
        if training_df[col].dtype in ['int64', 'float64']:
            non_null_pct = (training_df[col].notna().sum() / len(training_df)) * 100
            logger.info(f"  {col}: {non_null_pct:.1f}% non-null")
    
    # Check for any completely empty features
    empty_features = [col for col in feature_cols if training_df[col].isna().all()]
    if empty_features:
        logger.warning(f"Features with all missing values: {empty_features}")


def merge_training_table(features_path: str, labels_path: str, output_path: str) -> None:
    """Merge features and labels into training table.
    
    Args:
        features_path: Path to features parquet file
        labels_path: Path to labels parquet file
        output_path: Path to save training table parquet file
    """
    start_time = time.time()
    logger = get_logger("merge_training_table")
    
    logger.info("Starting training table merge process...")
    logger.info(f"Features path: {features_path}")
    logger.info(f"Labels path: {labels_path}")
    logger.info(f"Output path: {output_path}")
    
    # Load features and labels
    logger.info("Loading features...")
    features = read_parquet(features_path)
    log_dataframe_info(logger, features, "Features")
    
    logger.info("Loading labels...")
    labels = read_parquet(labels_path)
    log_dataframe_info(logger, labels, "Labels")
    
    # Validate merge keys
    validate_merge_keys(features, labels, logger)
    
    # Merge datasets
    training_table = merge_features_labels(features, labels, logger)
    
    # Analyze final table
    analyze_training_table(training_table, logger)
    
    # Save training table
    logger.info(f"Saving training table to {output_path}")
    write_parquet(training_table, output_path)
    
    end_time = time.time()
    log_execution_time(logger, start_time, end_time, "Training table merge")
    
    logger.info("Training table merge completed successfully!")


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description="Merge features and labels into training table")
    parser.add_argument("--features", required=True, help="Path to features parquet file")
    parser.add_argument("--labels", required=True, help="Path to labels parquet file")
    parser.add_argument("--output", required=True, help="Output path for training table parquet file")
    
    args = parser.parse_args()
    
    merge_training_table(
        features_path=args.features,
        labels_path=args.labels,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
