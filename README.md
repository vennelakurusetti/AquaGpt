# AquaGpt

A lightweight groundwater prediction project with Python data processing and ML training logic.

## Project overview

- `SIH_Project/src/train.py`: loads groundwater telemetry data, preprocesses it, trains a `RandomForestRegressor`, evaluates performance, and saves a model.
- `SIH_Project/requirements.txt`: Python dependencies for FastAPI support and model training.
- `ground_water/`: contains groundwater sample CSV files used for analysis.
- `aquagpt-ml model.ipynb`: notebook for exploratory analysis or model experimentation.

## Current state

- This repository includes code and sample CSV files under the GitHub file size limit.
- Extra very large files (`model_v1.pkl`, extremely large CSV exports) were excluded from the remote push because GitHub rejects files larger than 100MB.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r SIH_Project/requirements.txt
```

## Training

Run the training script from the repository root:

```bash
cd SIH_Project
python src/train.py
```

This will:

- load `groundwater-level-collected-telemetric.csv`
- preprocess data
- save `data/normalized_groundwater_data.csv`
- train a Random Forest model
- save the model to `models/model_v1.pkl`

## Notes

- If the raw dataset is missing, place your groundwater telemetry CSV in the repository root or update the `data_path` variable in `SIH_Project/src/train.py`.
- The model file is large and is not tracked in the remote repository for GitHub compatibility.
