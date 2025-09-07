"""Compute rolling statistics features over 180-day lookback window."""

import argparse
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from common.io import read_csv, write_parquet, read_parquet
from common.logging import get_logger, log_dataframe_info, log_execution_time


def select_observations_by_loinc(observations_df: pd.DataFrame, loinc_codes: List[str]) -> pd.DataFrame:
    """Select observations by LOINC codes.
    
    Args:
        observations_df: DataFrame with observations
        loinc_codes: List of LOINC codes to filter by
        
    Returns:
        Filtered observations DataFrame
    """
    if not loinc_codes:
        return pd.DataFrame()
    
    return observations_df[observations_df["Code"].astype(str).isin(loinc_codes)]


def select_observations_by_description(observations_df: pd.DataFrame, search_term: str) -> pd.DataFrame:
    """Select observations by description text search.
    
    Args:
        observations_df: DataFrame with observations
        search_term: Term to search for in description
        
    Returns:
        Filtered observations DataFrame
    """
    return observations_df[
        observations_df["Description"].str.contains(search_term, case=False, na=False)
    ]


def get_last_numeric_value(series: pd.Series) -> float:
    """Get the last numeric value from a series.
    
    Args:
        series: Pandas series with values
        
    Returns:
        Last numeric value or NaN if none found
    """
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    return numeric_series.iloc[-1] if len(numeric_series) > 0 else np.nan


def get_numeric_stats(series: pd.Series) -> Tuple[float, float]:
    """Get mean and standard deviation of numeric values.
    
    Args:
        series: Pandas series with values
        
    Returns:
        Tuple of (mean, std) or (NaN, NaN) if no numeric values
    """
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric_series) == 0:
        return np.nan, np.nan
    
    return float(numeric_series.mean()), float(numeric_series.std(ddof=0))


def compute_patient_features(
    patient_id: str,
    as_of_date: pd.Timestamp,
    lookback_days: int,
    observations_df: pd.DataFrame,
    encounters_df: pd.DataFrame,
    medications_df: pd.DataFrame,
    hba1c_codes: List[str],
    sbp_codes: List[str],
    dbp_codes: List[str],
    logger
) -> dict:
    """Compute features for a single patient.
    
    Args:
        patient_id: Patient identifier
        as_of_date: As-of date for feature computation
        lookback_days: Number of days to look back
        observations_df: Observations data
        encounters_df: Encounters data
        medications_df: Medications data
        hba1c_codes: LOINC codes for HbA1c
        sbp_codes: LOINC codes for systolic BP
        dbp_codes: LOINC codes for diastolic BP
        logger: Logger instance
        
    Returns:
        Dictionary with computed features
    """
    # Define lookback window
    lookback_start = as_of_date - pd.Timedelta(days=lookback_days)
    
    # Filter data for this patient and time window
    patient_obs = observations_df[
        (observations_df["Patient"] == patient_id) &
        (observations_df["Date"].between(lookback_start, as_of_date))
    ]
    
    patient_enc = encounters_df[
        (encounters_df["Patient"] == patient_id) &
        (encounters_df["Start"].between(lookback_start, as_of_date))
    ]
    
    # For medications, include those that were active during the window
    patient_meds = medications_df[
        (medications_df["Patient"] == patient_id) &
        (medications_df["Start"] <= as_of_date) &
        ((medications_df["Stop"].isna()) | (medications_df["Stop"] >= lookback_start))
    ]
    
    # Basic counts
    n_observations = len(patient_obs)
    n_encounters = len(patient_enc)
    n_active_meds = len(patient_meds["Description"].dropna().unique())
    
    # HbA1c features
    if hba1c_codes:
        hba1c_obs = select_observations_by_loinc(patient_obs, hba1c_codes)
    else:
        hba1c_obs = select_observations_by_description(patient_obs, "A1c")
    
    hba1c_obs = hba1c_obs.sort_values("Date")
    hba1c_last = get_last_numeric_value(hba1c_obs["Value"])
    hba1c_mean, hba1c_std = get_numeric_stats(hba1c_obs["Value"])
    
    # Blood pressure features
    if sbp_codes:
        sbp_obs = select_observations_by_loinc(patient_obs, sbp_codes)
    else:
        sbp_obs = select_observations_by_description(patient_obs, "systolic")
    
    if dbp_codes:
        dbp_obs = select_observations_by_loinc(patient_obs, dbp_codes)
    else:
        dbp_obs = select_observations_by_description(patient_obs, "diastolic")
    
    sbp_last = get_last_numeric_value(sbp_obs.sort_values("Date")["Value"])
    dbp_last = get_last_numeric_value(dbp_obs.sort_values("Date")["Value"])
    
    return {
        "patient": patient_id,
        "as_of": as_of_date,
        "n_observations_180d": n_observations,
        "n_encounters_180d": n_encounters,
        "n_active_meds_180d": n_active_meds,
        "hba1c_last": hba1c_last,
        "hba1c_mean": hba1c_mean,
        "hba1c_std": hba1c_std,
        "sbp_last": sbp_last,
        "dbp_last": dbp_last
    }


def compute_rolling_features(
    input_dir: str,
    cohort_path: str,
    lookback_days: int,
    hba1c_codes: str,
    sbp_codes: str,
    dbp_codes: str,
    output_path: str
) -> None:
    """Compute rolling statistics features for the cohort.
    
    Args:
        input_dir: Directory containing Synthea CSV files
        cohort_path: Path to cohort parquet file
        lookback_days: Number of days to look back for features
        hba1c_codes: Comma-separated LOINC codes for HbA1c
        sbp_codes: Comma-separated LOINC codes for systolic BP
        dbp_codes: Comma-separated LOINC codes for diastolic BP
        output_path: Path to save features parquet file
    """
    start_time = time.time()
    logger = get_logger("rolling_stats")
    
    logger.info("Starting feature computation process...")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Cohort path: {cohort_path}")
    logger.info(f"Lookback days: {lookback_days}")
    logger.info(f"Output path: {output_path}")
    
    # Parse LOINC codes
    hba1c_code_list = [c.strip() for c in hba1c_codes.split(",")] if hba1c_codes else []
    sbp_code_list = [c.strip() for c in sbp_codes.split(",")] if sbp_codes else []
    dbp_code_list = [c.strip() for c in dbp_codes.split(",")] if dbp_codes else []
    
    logger.info(f"HbA1c codes: {hba1c_code_list}")
    logger.info(f"SBP codes: {sbp_code_list}")
    logger.info(f"DBP codes: {dbp_code_list}")
    
    # Load cohort
    logger.info("Loading cohort data...")
    cohort = read_parquet(cohort_path)
    log_dataframe_info(logger, cohort, "Cohort")
    
    # Load clinical data
    logger.info("Loading observations data...")
    observations = read_csv(os.path.join(input_dir, "observations.csv"), parse_dates=["Date"])
    log_dataframe_info(logger, observations, "Observations")
    
    logger.info("Loading encounters data...")
    encounters = read_csv(os.path.join(input_dir, "encounters.csv"), parse_dates=["Start", "Stop"])
    log_dataframe_info(logger, encounters, "Encounters")
    
    logger.info("Loading medications data...")
    medications = read_csv(os.path.join(input_dir, "medications.csv"), parse_dates=["Start", "Stop"])
    log_dataframe_info(logger, medications, "Medications")
    
    # Compute features for each patient
    logger.info("Computing features for each patient...")
    features = []
    total_patients = len(cohort)
    
    for idx, (patient_id, as_of_date) in enumerate(cohort[["patient", "as_of"]].itertuples(index=False)):
        if idx % 50 == 0:
            logger.info(f"Processing patient {idx + 1}/{total_patients}")
        
        patient_features = compute_patient_features(
            patient_id=patient_id,
            as_of_date=as_of_date,
            lookback_days=lookback_days,
            observations_df=observations,
            encounters_df=encounters,
            medications_df=medications,
            hba1c_codes=hba1c_code_list,
            sbp_codes=sbp_code_list,
            dbp_codes=dbp_code_list,
            logger=logger
        )
        
        features.append(patient_features)
    
    # Create features DataFrame
    features_df = pd.DataFrame(features)
    
    # Log feature statistics
    logger.info("Feature computation completed. Summary statistics:")
    for col in features_df.select_dtypes(include=[np.number]).columns:
        if col not in ["patient"]:
            non_null_count = features_df[col].notna().sum()
            logger.info(f"{col}: {non_null_count}/{len(features_df)} non-null values")
    
    # Save features
    logger.info(f"Saving features to {output_path}")
    write_parquet(features_df, output_path)
    
    log_dataframe_info(logger, features_df, "Final features")
    
    end_time = time.time()
    log_execution_time(logger, start_time, end_time, "Feature computation")
    
    logger.info("Feature computation completed successfully!")


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description="Compute rolling statistics features")
    parser.add_argument("--input-dir", required=True, help="Directory containing Synthea CSV files")
    parser.add_argument("--cohort", required=True, help="Path to cohort parquet file")
    parser.add_argument("--lookback-days", type=int, required=True, help="Lookback window in days")
    parser.add_argument("--hba1c-codes", default="", help="Comma-separated LOINC codes for HbA1c")
    parser.add_argument("--sbp-codes", default="", help="Comma-separated LOINC codes for systolic BP")
    parser.add_argument("--dbp-codes", default="", help="Comma-separated LOINC codes for diastolic BP")
    parser.add_argument("--output", required=True, help="Output path for features parquet file")
    
    args = parser.parse_args()
    
    compute_rolling_features(
        input_dir=args.input_dir,
        cohort_path=args.cohort,
        lookback_days=args.lookback_days,
        hba1c_codes=args.hba1c_codes,
        sbp_codes=args.sbp_codes,
        dbp_codes=args.dbp_codes,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
