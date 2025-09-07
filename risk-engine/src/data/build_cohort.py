"""Build cohort of adult diabetic patients with assigned as-of dates."""

import argparse
import os
import time
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from common.io import read_csv, write_parquet
from common.logging import get_logger, log_dataframe_info, log_execution_time
from common.config import load_config


# SNOMED codes for diabetes (broad definition)
DIABETES_SNOMED = {"44054006"}  # Diabetes mellitus


def age_on(birthdate: pd.Timestamp, on_date: pd.Timestamp) -> int:
    """Calculate age on a specific date.
    
    Args:
        birthdate: Birth date
        on_date: Date to calculate age on
        
    Returns:
        Age in years
    """
    return on_date.year - birthdate.year - (
        (on_date.month, on_date.day) < (birthdate.month, birthdate.day)
    )


def identify_diabetes_patients(conditions_df: pd.DataFrame, logger) -> pd.DataFrame:
    """Identify patients with diabetes from conditions data.
    
    Args:
        conditions_df: DataFrame with conditions data
        logger: Logger instance
        
    Returns:
        DataFrame with unique patient IDs who have diabetes
    """
    logger.info("Identifying diabetes patients...")
    
    # Filter for diabetes conditions
    diabetes_conditions = conditions_df[
        conditions_df["Code"].astype(str).isin(DIABETES_SNOMED)
    ]
    
    logger.info(f"Found {len(diabetes_conditions)} diabetes condition records")
    
    # Get unique patients with diabetes
    diabetes_patients = diabetes_conditions[["Patient"]].drop_duplicates()
    diabetes_patients["diabetes"] = 1
    
    logger.info(f"Identified {len(diabetes_patients)} unique diabetes patients")
    
    return diabetes_patients


def filter_by_age(patients_df: pd.DataFrame, ref_date: pd.Timestamp, min_age: int, logger) -> pd.DataFrame:
    """Filter patients by minimum age on reference date.
    
    Args:
        patients_df: DataFrame with patient data
        ref_date: Reference date for age calculation
        min_age: Minimum age threshold
        logger: Logger instance
        
    Returns:
        DataFrame with age-filtered patients
    """
    logger.info(f"Filtering patients by minimum age {min_age} on {ref_date.date()}")
    
    # Calculate age on reference date
    patients_df["age"] = patients_df["BirthDate"].apply(
        lambda d: age_on(pd.to_datetime(d), ref_date)
    )
    
    # Filter by minimum age
    age_filtered = patients_df[patients_df["age"] >= min_age]
    
    logger.info(f"Patients after age filter: {len(age_filtered)} (removed {len(patients_df) - len(age_filtered)})")
    
    return age_filtered


def build_cohort(input_dir: str, reference_date: str, min_age: int, output_path: str) -> None:
    """Build cohort of adult diabetic patients.
    
    Args:
        input_dir: Directory containing Synthea CSV files
        reference_date: Reference date for cohort (YYYY-MM-DD)
        min_age: Minimum age for inclusion
        output_path: Path to save cohort parquet file
    """
    start_time = time.time()
    logger = get_logger("build_cohort")
    
    logger.info("Starting cohort building process...")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Reference date: {reference_date}")
    logger.info(f"Minimum age: {min_age}")
    logger.info(f"Output path: {output_path}")
    
    # Parse reference date
    ref_date = pd.to_datetime(reference_date)
    
    # Load data
    logger.info("Loading patients data...")
    patients_path = os.path.join(input_dir, "patients.csv")
    patients = read_csv(patients_path, parse_dates=["BirthDate", "DeathDate"])
    log_dataframe_info(logger, patients, "Patients")
    
    logger.info("Loading conditions data...")
    conditions_path = os.path.join(input_dir, "conditions.csv")
    conditions = read_csv(conditions_path, parse_dates=["Start", "Stop"])
    log_dataframe_info(logger, conditions, "Conditions")
    
    # Identify diabetes patients
    diabetes_patients = identify_diabetes_patients(conditions, logger)
    
    # Merge patients with diabetes flag
    patients_with_diabetes = patients.merge(
        diabetes_patients, 
        how="inner", 
        left_on="Id", 
        right_on="Patient"
    )
    
    logger.info(f"Patients with diabetes: {len(patients_with_diabetes)}")
    
    # Filter by age
    age_filtered_patients = filter_by_age(patients_with_diabetes, ref_date, min_age, logger)
    
    # Create final cohort
    cohort = age_filtered_patients[["Id", "age"]].rename(columns={"Id": "patient"})
    cohort["as_of"] = ref_date
    
    # Reorder columns
    cohort = cohort[["patient", "age", "as_of"]]
    
    # Save cohort
    logger.info(f"Saving cohort to {output_path}")
    write_parquet(cohort, output_path)
    
    log_dataframe_info(logger, cohort, "Final cohort")
    
    end_time = time.time()
    log_execution_time(logger, start_time, end_time, "Cohort building")
    
    logger.info(f"Cohort building completed successfully!")


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description="Build cohort of adult diabetic patients")
    parser.add_argument("--input-dir", required=True, help="Directory containing Synthea CSV files")
    parser.add_argument("--reference-date", required=True, help="Reference date (YYYY-MM-DD)")
    parser.add_argument("--min-age", type=int, default=18, help="Minimum age for inclusion")
    parser.add_argument("--output", required=True, help="Output path for cohort parquet file")
    
    args = parser.parse_args()
    
    build_cohort(
        input_dir=args.input_dir,
        reference_date=args.reference_date,
        min_age=args.min_age,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
