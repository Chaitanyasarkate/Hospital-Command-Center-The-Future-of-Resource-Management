"""
staff_load.py
Compute staff workload ratio and produce classification and recommendations.

Functions:
 - compute_load_and_recommendations(predicted_er_list, df): returns aggregated workload status and recommendations
"""
import numpy as np
import pandas as pd


def classify_ratio(ratio):
    """Classify patient-to-staff ratio:
    - <1.5: Normal
    - 1.5-2.0: High
    - >2.0: Critical
    """
    if ratio < 1.5:
        return 'Normal'
    elif ratio <= 2.0:
        return 'High'
    else:
        return 'Critical'


def compute_load_and_recommendations(predicted_er_list, df):
    """Aggregate predicted ER admissions to estimate staff workload and recommendations.

    Uses median staff_on_duty from historical df to estimate available staff.
    Returns dict with keys: ratio, status, recommendations
    """
    # Sum predicted ER over 24h as rough patient load; alternatively use max for peaks
    predicted_total = float(np.sum(predicted_er_list))

    median_staff = float(df['staff_on_duty'].median())
    # patients per staff ratio
    ratio = predicted_total / max(1.0, median_staff)

    status = classify_ratio(ratio)

    recommendations = []
    if status == 'High':
        recommendations.append('Consider adding extra nursing shifts')
    elif status == 'Critical':
        recommendations.append('Activate emergency staffing protocol: call standby staff and cancel non-critical leave')
        recommendations.append('Redistribute non-critical patients and consider tele-triage')

    # Additional recommendation based on ICU predictions will be combined upstream
    return {
        'predicted_total_er_24h': predicted_total,
        'median_staff': median_staff,
        'ratio': round(ratio, 2),
        'status': status,
        'recommendations': recommendations,
    }
