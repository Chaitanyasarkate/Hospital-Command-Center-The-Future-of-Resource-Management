"""
icu_prediction.py
Predict ICU bed demand from ER admissions using Linear Regression.

Functions:
 - predict_icu_demand(csv_path, er_predictions): trains on historical er_admissions -> icu_beds_used
   and returns predicted icu beds for the provided er_predictions list.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'icu_model.joblib')


def predict_icu_demand(csv_path: str, er_predictions, force_retrain: bool = False):
    """Train or load a regression mapping ER admissions to ICU beds and predict for er_predictions.

    If a persisted model exists it will be loaded unless force_retrain=True.
    """
    df = pd.read_csv(csv_path)
    # Accept flexible ER column names
    if 'er_admissions' in df.columns:
        er_col = 'er_admissions'
    elif 'emergency_patients' in df.columns:
        er_col = 'emergency_patients'
    else:
        raise KeyError("No ER admissions column found in CSV for ICU model.")

    if 'icu_beds_used' not in df.columns:
        raise KeyError("No 'icu_beds_used' column found in CSV for ICU model.")

    X = df[[er_col]].values
    y = df['icu_beds_used'].values

    # Load persisted model when available
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
            pass

    er_array = np.array(er_predictions).reshape(-1, 1)
    preds = model.predict(er_array)
    preds = [max(0.0, float(x)) for x in preds]
    return preds


if __name__ == '__main__':
    demo_er = [10, 12, 15]
    print(predict_icu_demand('data/hospital_data.csv', demo_er))
