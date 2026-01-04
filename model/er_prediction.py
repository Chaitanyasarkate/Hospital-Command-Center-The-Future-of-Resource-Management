"""
er_prediction.py
Simple, explainable ER admissions prediction using Linear Regression.

Functions:
 - predict_next_24_hours(csv_path): returns a list of 24 hourly ER admission predictions

The model uses features: hour of day, day of week, temperature, and previous-hour admissions (lag-1).
"""
from datetime import timedelta
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

# persisted model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'er_model.joblib')


def _prepare_features(df):
    df = df.copy()
    # Accept flexible date column names
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    elif 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'])
    elif 'Date' in df.columns:
        df['datetime'] = pd.to_datetime(df['Date'])
    else:
        raise KeyError("No date/datetime column found in CSV. Expected 'datetime' or 'date'.")

    df['hour'] = df['datetime'].dt.hour
    df['dow'] = df['datetime'].dt.dayofweek

    # Accept flexible ER column names and create a canonical 'er_admissions'
    if 'er_admissions' not in df.columns:
        if 'emergency_patients' in df.columns:
            df['er_admissions'] = df['emergency_patients']
        elif 'er' in df.columns:
            df['er_admissions'] = df['er']
        else:
            raise KeyError("No ER admissions column found. Expected 'er_admissions' or 'emergency_patients'.")

    # Ensure temperature exists (use median or default if missing)
    if 'temperature' not in df.columns:
        try:
            temp_val = float(df.get('temperature', np.nan).median())
        except Exception:
            temp_val = 25.0
        df['temperature'] = temp_val

    # lag-1 ER admissions (backfill the first value)
    df['lag1_er'] = df['er_admissions'].shift(1).bfill()
    return df


def predict_next_24_hours(csv_path: str, force_retrain: bool = False):
    """Train a simple Linear Regression and predict next 24 hourly ER admissions.

    Returns: list of 24 floats (predicted er_admissions)
    """
    df = pd.read_csv(csv_path)
    df = _prepare_features(df)

    # Features and target
    features = ['hour', 'dow', 'temperature', 'lag1_er']
    X = df[features].values
    y = df['er_admissions'].values

    # Load persisted model if available, unless force_retrain requested
    if os.path.exists(MODEL_PATH) and not force_retrain:
        try:
            model = joblib.load(MODEL_PATH)
        except Exception:
            model = LinearRegression()
            model.fit(X, y)
            joblib.dump(model, MODEL_PATH)
    else:
        model = LinearRegression()
        model.fit(X, y)
        try:
            joblib.dump(model, MODEL_PATH)
        except Exception:
            # keep running even if saving fails
            pass

    # Predict iteratively for the next 24 hours using predicted lag
    last_row = df.iloc[-1]
    next_dt = pd.to_datetime(last_row['datetime']) + timedelta(hours=1)
    lag = float(last_row['er_admissions'])
    preds = []
    for i in range(24):
        hour = next_dt.hour
        dow = next_dt.dayofweek
        temp = float(last_row['temperature'])  # assume last known temperature; in real app use forecast
        x = np.array([[hour, dow, temp, lag]])
        p = model.predict(x)[0]
        p = max(0.0, float(p))
        preds.append(p)
        # update lag and time
        lag = p
        next_dt += timedelta(hours=1)

    return preds


if __name__ == '__main__':
    # quick demo when run directly
    preds = predict_next_24_hours('data/hospital_data.csv')
    print('Next 24h ER predictions:', [round(x,1) for x in preds])
