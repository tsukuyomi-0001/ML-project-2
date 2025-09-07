# COVID-19 Death Prediction using XGBoost

This project predicts the number of COVID-19 related deaths based on time-series data aggregated by country and state.  
The dataset includes daily records of confirmed cases, recoveries, and deaths along with geospatial and temporal attributes.

---

## Project Overview

- **Goal:** Predict daily COVID-19 deaths for each region (country/state).
- **Approach:** 
  - Feature engineering on temporal and categorical data.
  - Training an XGBoost regressor for regression task.
  - Model evaluated using Root Mean Squared Error (RMSE).

---

## Dataset

- Columns:
  - `Date` (converted to Year, Month, Day)
  - `Country`
  - `State`
  - `Confirmed`
  - `Recovered`
  - `Deaths` (Target)

- **Excluded Columns:**
  - `Lat`, `Long` were excluded to avoid overfitting and noise.
  
- Dataset split: **Train / Validation / Test**

---

## Pipeline

- **Preprocessing:**
  - Categorical features: One-hot encoding
  - Numerical features: Imputation (most frequent)
  - Temporal feature: Extracted Year, Month, Day from `Date`

- **Model:**
  - XGBoost Regressor
  - Tuned with manual adjustments (RandomizedSearchCV not applied due to mixed-country time-series nature)

- **Evaluation Metric:** Root Mean Squared Error (RMSE)

---

## Results

- **Validation RMSE:** ~2191  
- **Test RMSE:** ~3312  

- Log-transform on target (`log1p`) was tested but did not outperform the raw scale.

---

## Files

- `main.ipynb` – Full project notebook with EDA, preprocessing, and model training.
- `dataset.csv` – Source dataset (not included here due to licensing/size).
- `requirements.txt` – Python dependencies.

---

## How to Run

1. Clone this repository:
   ```bash
   git clone <repo_url>
   cd covid19-death-prediction
    ```

3. Requirement Install:
    ```bash
    pip install -r requirement.txt
    ```
2. To run:
    ```bash
    python process.py
    ```