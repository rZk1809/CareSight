# CareSight Risk Engine

An explainable ML pipeline that predicts 90-day deterioration risk for chronic-care patients using 30–180 days of history.

## Overview

This project builds a complete machine learning pipeline for healthcare risk prediction with:
- Cohort identification and labeling
- Feature engineering from clinical data
- LightGBM model training with calibration
- Comprehensive evaluation and explainability
- DVC-orchestrated reproducible pipeline

## Quick Start

1. **Setup Environment**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Data Source**
   Edit `configs/data.yaml` to point to your Synthea CSV output directory.

3. **Run Pipeline**
   ```bash
   make all
   # or
   dvc repro
   ```

## Project Structure

```
risk-engine/
├── data/              # DVC-tracked outputs & artifacts
├── configs/           # Configuration files
├── src/               # Source code modules
├── notebooks/         # Jupyter notebooks for analysis
├── tests/             # Unit tests
├── scripts/           # Utility scripts
└── docs/              # Documentation
```

## Pipeline Stages

1. **Build Cohort** - Select adult diabetic patients
2. **Make Labels** - Identify 90-day deterioration events
3. **Feature Engineering** - Compute 180-day rolling statistics
4. **Train Model** - LightGBM classifier with calibration
5. **Evaluate** - Metrics, confusion matrices, and reports

## Requirements

- Python 3.8+
- Synthea-generated CSV data
- DVC for pipeline orchestration
