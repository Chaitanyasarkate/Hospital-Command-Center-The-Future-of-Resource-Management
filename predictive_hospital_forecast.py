"""
Predictive Hospital Resource & Staff Workload Forecasting (Linear Regression)

This single-file project is beginner-friendly and demonstrates:
 - Loading and preprocessing hospital data from CSV
 - Training three linear regression models:
     Model 1: Predict total_patients
     Model 2: Predict icu_beds_used (from predicted total_patients)
     Model 3: Predict staff_workload_index (from predicted total_patients and staff_on_duty)
 - Making a next-month prediction (assumes next month = last month + 1)
 - Printing clear outputs and simple hospital alert logic

Requirements: pandas, numpy, scikit-learn

Usage:
 - Place `hospital_past_2_months_data.csv` in the same directory as this script
 - Run: python predictive_hospital_forecast.py

Author: Senior AI/ML engineer (example for final-year project)
"""

import os
from datetime import datetime

try:
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
except ImportError as e:
    raise ImportError(
        f"A required package is missing: {e}.\nPlease install dependencies: pip install -r requirements.txt"
    )


def load_data(csv_path: str) -> pd.DataFrame:
    """Load CSV into a pandas DataFrame and perform basic validation."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the hospital data:
    - Convert `date` to datetime
    - Extract `day` and `month` features
    - Handle missing values (forward-fill then median for numerical columns)
    """
    df = df.copy()

    # Convert date column to datetime. If conversion fails, try common formats.
    try:
        df['date'] = pd.to_datetime(df['date'])
    except Exception:
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')

    # Extract day and month as numeric features
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month

    # Fill missing dates (unlikely) by forward fill
    df['date'] = df['date'].ffill()

    # Identify numeric columns for median imputation fallback
    numeric_cols = ['total_beds', 'icu_beds_used', 'emergency_patients', 'normal_beds_used',
                    'total_patients', 'staff_on_duty', 'staff_workload_index']

    # Forward fill then median for remaining NaNs in numeric columns
    df[numeric_cols] = df[numeric_cols].ffill()
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # As a final safety, drop any rows that still contain NaNs
    df = df.dropna().reset_index(drop=True)

    return df


def train_model_total_patients(df: pd.DataFrame) -> LinearRegression:
    """Train Model 1 to predict total_patients.
    Inputs: day, month, icu_beds_used, emergency_patients, staff_on_duty
    """
    features = ['day', 'month', 'icu_beds_used', 'emergency_patients', 'staff_on_duty']
    X = df[features].values
    y = df['total_patients'].values

    # Simple train/test split to validate model performance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate and print a simple metric (R^2)
    y_pred_test = model.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    print(f"Model 1 (total_patients) trained -- R^2 on holdout set: {r2:.3f}")

    return model


def train_model_icu_from_total(df: pd.DataFrame) -> LinearRegression:
    """Train Model 2 to predict icu_beds_used from total_patients.
    Input: total_patients
    """
    X = df[['total_patients']].values
    y = df['icu_beds_used'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    print(f"Model 2 (icu_beds_used) trained -- R^2 on holdout set: {r2:.3f}")

    return model


def train_model_staff_workload(df: pd.DataFrame) -> LinearRegression:
    """Train Model 3 to predict staff_workload_index.
    Inputs: total_patients, staff_on_duty
    """
    features = ['total_patients', 'staff_on_duty']
    X = df[features].values
    y = df['staff_workload_index'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    print(f"Model 3 (staff_workload_index) trained -- R^2 on holdout set: {r2:.3f}")

    return model


def compute_next_month_sample_inputs(df: pd.DataFrame) -> dict:
    """Create reasonable sample inputs for the next month prediction.
    - next month = last month + 1 (wraps to 1 after 12)
    - day: use the median day from the data as a representative day
    - icu_beds_used, emergency_patients, staff_on_duty: use recent averages
    Returns a dict with keys matching model 1 inputs.
    """
    last_month = int(df['month'].iloc[-1])
    next_month = last_month + 1
    if next_month > 12:
        next_month = 1

    # Use the median day (representative) to avoid edge-case dates
    sample_day = int(df['day'].median())

    # Use rolling averages (last 7 days if available) to simulate expected input
    window = min(7, len(df))
    recent = df.tail(window)

    sample_icu_beds_used = float(recent['icu_beds_used'].mean())
    sample_emergency_patients = float(recent['emergency_patients'].mean())
    sample_staff_on_duty = float(recent['staff_on_duty'].mean())

    return {
        'day': sample_day,
        'month': next_month,
        'icu_beds_used': sample_icu_beds_used,
        'emergency_patients': sample_emergency_patients,
        'staff_on_duty': sample_staff_on_duty,
    }


def hospital_alert_logic(predicted_icu_beds: float, icu_capacity: float) -> str:
    """Return alert level string based on ICU usage relative to capacity.
    - HIGH LOAD if ICU usage > 80% of capacity
    - MEDIUM LOAD if ICU usage > 60% of capacity
    - NORMAL otherwise
    """
    if icu_capacity <= 0:
        return "UNKNOWN (invalid ICU capacity)"

    usage_pct = (predicted_icu_beds / icu_capacity) * 100.0

    if usage_pct > 80.0:
        return f"HIGH LOAD ({usage_pct:.1f}% of ICU capacity)"
    elif usage_pct > 60.0:
        return f"MEDIUM LOAD ({usage_pct:.1f}% of ICU capacity)"
    else:
        return f"NORMAL ({usage_pct:.1f}% of ICU capacity)"


def main():
    # File path (expect the CSV in the same directory)
    csv_filename = 'hospital_past_2_months_data.csv'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, csv_filename)

    print('\n--- Predictive Hospital Resource & Staff Workload Forecasting ---\n')

    # 1) Load data
    try:
        raw_df = load_data(csv_path)
    except FileNotFoundError as e:
        print(e)
        print('\nPlease place the CSV file in the same directory as this script and re-run.')
        return

    # 2) Preprocess
    df = preprocess_data(raw_df)

    # Basic check: ensure there is data
    if df.shape[0] < 10:
        print('Warning: dataset seems small (less than 10 rows). Models may be unreliable.')

    # 3) Train models
    model_total_patients = train_model_total_patients(df)
    model_icu_from_total = train_model_icu_from_total(df)
    model_staff_workload = train_model_staff_workload(df)

    # 4) Prepare sample inputs for next month prediction
    sample_inputs = compute_next_month_sample_inputs(df)

    # Model 1 prediction: predict total_patients for next month
    model1_features = np.array([[
        sample_inputs['day'],
        sample_inputs['month'],
        sample_inputs['icu_beds_used'],
        sample_inputs['emergency_patients'],
        sample_inputs['staff_on_duty'],
    ]])

    predicted_total_patients = model_total_patients.predict(model1_features)[0]
    predicted_total_patients = max(predicted_total_patients, 0.0)  # no negative patients

    # Model 2 prediction: predict icu_beds_used from predicted total_patients
    model2_features = np.array([[predicted_total_patients]])
    predicted_icu_beds = model_icu_from_total.predict(model2_features)[0]
    predicted_icu_beds = max(predicted_icu_beds, 0.0)

    # Model 3 prediction: predict staff_workload_index from predicted total_patients and staff_on_duty
    model3_features = np.array([[predicted_total_patients, sample_inputs['staff_on_duty']]])
    predicted_staff_workload = model_staff_workload.predict(model3_features)[0]

    # 5) ICU capacity assumption
    # There is no explicit ICU capacity column. We make a reasonable assumption:
    # - Use the last observed `total_beds` and assume a fraction is ICU capacity.
    #   Many hospitals allocate roughly 10-20% of beds to ICU; we'll conservatively assume 15%.
    last_total_beds = float(df['total_beds'].iloc[-1])
    icu_capacity_assumed = last_total_beds * 0.15

    # Ensure capacity is at least the historical max icu usage to avoid absurd alerts
    historical_max_icu = float(df['icu_beds_used'].max())
    icu_capacity = max(icu_capacity_assumed, historical_max_icu * 1.05)

    # 6) Alert logic
    alert = hospital_alert_logic(predicted_icu_beds, icu_capacity)

    # 7) Print clear, formatted outputs
    print('Next-month prediction inputs (representative):')
    print(f" - day: {sample_inputs['day']}")
    print(f" - month: {sample_inputs['month']}")
    print(f" - icu_beds_used (recent avg): {sample_inputs['icu_beds_used']:.1f}")
    print(f" - emergency_patients (recent avg): {sample_inputs['emergency_patients']:.1f}")
    print(f" - staff_on_duty (recent avg): {sample_inputs['staff_on_duty']:.1f}\n")

    print('Predicted next-month metrics:')
    print(f" - Predicted total patients: {predicted_total_patients:.1f}")
    print(f" - Predicted ICU beds required: {predicted_icu_beds:.1f}")
    print(f" - Predicted staff workload index: {predicted_staff_workload:.2f}\n")

    print('ICU capacity (assumed):')
    print(f" - ICU capacity (assumed): {icu_capacity:.1f} beds")
    print(f" - Alert level: {alert}\n")

    print('Notes:')
    print(' - Models are simple linear regressions trained on the provided 2-month dataset.')
    print(' - Results are illustrative; for production use, include more historical data, feature engineering, and uncertainty estimates.')


if __name__ == '__main__':
    main()
