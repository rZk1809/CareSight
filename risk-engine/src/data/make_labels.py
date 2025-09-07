"""Create 90-day deterioration labels for the cohort."""

import argparse
import os
import time
import pandas as pd
from pathlib import Path
from typing import List

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from common.io import read_csv, write_parquet, read_parquet
from common.logging import get_logger, log_dataframe_info, log_execution_time


def create_deterioration_labels(
    cohort_df: pd.DataFrame,
    encounters_df: pd.DataFrame,
    prediction_window_days: int,
    positive_classes: List[str],
    logger
) -> pd.DataFrame:
    """Create deterioration labels based on future encounters.
    
    Args:
        cohort_df: DataFrame with cohort (patient, as_of)
        encounters_df: DataFrame with encounter data
        prediction_window_days: Number of days to look ahead for deterioration
        positive_classes: List of encounter classes that indicate deterioration
        logger: Logger instance
        
    Returns:
        DataFrame with labels (patient, as_of, label_90d)
    """
    logger.info("Creating deterioration labels...")
    logger.info(f"Prediction window: {prediction_window_days} days")
    logger.info(f"Positive encounter classes: {positive_classes}")
    
    # Normalize encounter classes to lowercase for comparison
    pos_classes_lower = set([c.lower() for c in positive_classes])
    encounters_df["EncounterClass"] = encounters_df["EncounterClass"].str.lower()
    
    # Calculate prediction window
    pred_window = pd.Timedelta(days=prediction_window_days)
    
    labels = []
    total_patients = len(cohort_df)
    
    for idx, (patient_id, as_of_date) in enumerate(cohort_df[["patient", "as_of"]].itertuples(index=False)):
        if idx % 100 == 0:
            logger.info(f"Processing patient {idx + 1}/{total_patients}")
        
        # Define future window: (as_of, as_of + prediction_window_days]
        future_start = as_of_date
        future_end = as_of_date + pred_window
        
        # Get encounters for this patient in the future window
        patient_encounters = encounters_df[
            (encounters_df["Patient"] == patient_id) &
            (encounters_df["Start"] > future_start) &
            (encounters_df["Start"] <= future_end)
        ]
        
        # Check if any encounter has a positive class
        has_deterioration = patient_encounters["EncounterClass"].isin(pos_classes_lower).any()
        label = int(has_deterioration)
        
        labels.append({
            "patient": patient_id,
            "as_of": as_of_date,
            "label_90d": label
        })
    
    labels_df = pd.DataFrame(labels)
    
    # Log label distribution
    label_counts = labels_df["label_90d"].value_counts().sort_index()
    logger.info(f"Label distribution: {label_counts.to_dict()}")
    positive_rate = labels_df["label_90d"].mean()
    logger.info(f"Positive rate: {positive_rate:.3f} ({positive_rate*100:.1f}%)")
    
    return labels_df


def make_labels(
    input_dir: str,
    cohort_path: str,
    prediction_window_days: int,
    positive_classes: str,
    output_path: str
) -> None:
    """Create deterioration labels for the cohort.
    
    Args:
        input_dir: Directory containing Synthea CSV files
        cohort_path: Path to cohort parquet file
        prediction_window_days: Number of days to look ahead
        positive_classes: Comma-separated list of positive encounter classes
        output_path: Path to save labels parquet file
    """
    start_time = time.time()
    logger = get_logger("make_labels")
    
    logger.info("Starting label creation process...")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Cohort path: {cohort_path}")
    logger.info(f"Prediction window: {prediction_window_days} days")
    logger.info(f"Positive classes: {positive_classes}")
    logger.info(f"Output path: {output_path}")
    
    # Parse positive classes
    positive_class_list = [c.strip() for c in positive_classes.split(",")]
    
    # Load cohort
    logger.info("Loading cohort data...")
    cohort = read_parquet(cohort_path)
    log_dataframe_info(logger, cohort, "Cohort")
    
    # Load encounters
    logger.info("Loading encounters data...")
    encounters_path = os.path.join(input_dir, "encounters.csv")
    encounters = read_csv(encounters_path, parse_dates=["Start", "Stop"])
    log_dataframe_info(logger, encounters, "Encounters")
    
    # Log encounter class distribution
    encounter_classes = encounters["EncounterClass"].value_counts()
    logger.info(f"Encounter class distribution: {encounter_classes.to_dict()}")
    
    # Create labels
    labels = create_deterioration_labels(
        cohort_df=cohort,
        encounters_df=encounters,
        prediction_window_days=prediction_window_days,
        positive_classes=positive_class_list,
        logger=logger
    )
    
    # Save labels
    logger.info(f"Saving labels to {output_path}")
    write_parquet(labels, output_path)
    
    log_dataframe_info(logger, labels, "Final labels")
    
    end_time = time.time()
    log_execution_time(logger, start_time, end_time, "Label creation")
    
    logger.info("Label creation completed successfully!")


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description="Create 90-day deterioration labels")
    parser.add_argument("--input-dir", required=True, help="Directory containing Synthea CSV files")
    parser.add_argument("--cohort", required=True, help="Path to cohort parquet file")
    parser.add_argument("--prediction-window-days", type=int, required=True, help="Prediction window in days")
    parser.add_argument("--positive-classes", required=True, help="Comma-separated positive encounter classes")
    parser.add_argument("--output", required=True, help="Output path for labels parquet file")
    
    args = parser.parse_args()
    
    make_labels(
        input_dir=args.input_dir,
        cohort_path=args.cohort,
        prediction_window_days=args.prediction_window_days,
        positive_classes=args.positive_classes,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
