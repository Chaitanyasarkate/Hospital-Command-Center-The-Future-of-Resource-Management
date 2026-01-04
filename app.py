import os
import sys

# Dependency checks: give a clear message if required packages are missing
try:
    import pandas as pd
    import numpy as np
    from flask import Flask, render_template, make_response, request, redirect, url_for, jsonify
    from werkzeug.utils import secure_filename
except ImportError as e:
    print(f"A required package is missing: {e}\nPlease install dependencies with:\n    pip install -r requirements.txt")
    sys.exit(1)

# Import ML modules (these use scikit-learn)
try:
    from model.er_prediction import predict_next_24_hours
    from model.icu_prediction import predict_icu_demand
    from model.staff_load import compute_load_and_recommendations
except Exception as e:
    print(f"Error importing model modules: {e}\nMake sure scikit-learn is installed (pip install scikit-learn) and model files exist.")
    sys.exit(1)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, 'data', 'hospital_data.csv')
ALLOWED_EXT = {'csv'}
BED_HISTORY_CSV = os.path.join(BASE_DIR, 'data', 'bed_usage_history.csv')

app = Flask(__name__)
# Session secret key (required for flash/session). In production set SECRET_KEY env var.
import os as _os
app.secret_key = _os.environ.get('SECRET_KEY') or _os.urandom(24)

# --- logging setup -------------------------------------------------
import logging
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, 'app.log')
# Send logs to file only to avoid interleaving with Flask/Werkzeug console output
file_handler = logging.FileHandler(log_path)
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
logger = logging.getLogger('hospital-dashboard')
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
# Reduce Werkzeug's console noise (it logs requests at INFO); set it to WARNING
logging.getLogger('werkzeug').setLevel(logging.WARNING)
logger.propagate = False


def load_data():
    """Load historical hospital CSV and ensure datetime parsing."""
    df = pd.read_csv(DATA_CSV)
    # support files that use 'datetime' or 'date' column names
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    elif 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'])
    else:
        # raise with helpful message for the UI
        raise KeyError("CSV missing a 'datetime' or 'date' column. Please upload a CSV with a 'date' or 'datetime' column.")
    # Normalize ER admissions column: accept 'er_admissions' or 'emergency_patients'
    if 'er_admissions' not in df.columns:
        if 'emergency_patients' in df.columns:
            df['er_admissions'] = df['emergency_patients']
        elif 'er' in df.columns:
            df['er_admissions'] = df['er']
        else:
            # leave it to the caller to handle missing ER column; provide a helpful message
            raise KeyError("CSV missing ER admissions column. Expected 'er_admissions' or 'emergency_patients'.")

    # Ensure numeric columns are numeric
    for col in ['er_admissions']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def validate_csv_file(path: str):
    """Quickly validate an uploaded CSV for required columns before replacing the live dataset.

    Returns: (True, '') on success or (False, 'error message') on failure.
    """
    try:
        df = pd.read_csv(path, nrows=10)
    except Exception as e:
        return False, f'Uploaded file is not a readable CSV: {e}'

    cols = set(df.columns.str.lower()) if hasattr(df.columns, 'str') else set(df.columns)
    # require a date/datetime column
    if not ({'datetime', 'date'} & cols) and not ({'date', 'datetime', 'date'} & cols):
        # Accept 'Date' as well
        if 'date' not in cols and 'datetime' not in cols and 'date' not in cols:
            return False, "CSV missing a 'date' or 'datetime' column."

    # require an ER admissions column (flexible names)
    if not ({'er_admissions', 'emergency_patients', 'er'} & cols):
        return False, "CSV missing ER admissions column. Expected 'er_admissions' or 'emergency_patients'."

    return True, ''


@app.route('/preview', methods=['POST'])
def preview_csv():
    """Return a small preview of the uploaded CSV without installing it.

    Expects a multipart POST with field 'csv'. Returns JSON with detected columns
    and up to 10 sample rows.
    """
    if 'csv' not in request.files:
        return jsonify({'ok': False, 'message': 'No file part in request'}), 400
    file = request.files['csv']
    if file.filename == '':
        return jsonify({'ok': False, 'message': 'No file selected'}), 400

    import io
    try:
        # read a sample into pandas from the file stream
        stream = io.StringIO(file.stream.read().decode('utf-8', errors='replace'))
        df = pd.read_csv(stream, nrows=10)
    except Exception as e:
        logger.exception('Preview: failed to read uploaded CSV')
        return jsonify({'ok': False, 'message': f'Failed to read CSV: {e}'}), 400

    cols = list(df.columns)
    preview_rows = df.head(10).fillna('').to_dict(orient='records')

    # run quick validation
    ok, err = validate_csv_file(file.filename) if False else (True, '')
    # Note: we don't validate by path here; instead we'll return columns and let client decide

    return jsonify({'ok': True, 'columns': cols, 'preview': preview_rows, 'message': ''})


def generate_dashboard_data():
    """Run the three models and assemble dashboard data (predictions, alerts, recommendations)."""
    try:
        df = load_data()
    except Exception as e:
        # return a minimal dashboard with the error message so the UI can display it
        default_staff = {'predicted_total_er_24h': 0.0, 'median_staff': 0.0, 'ratio': 0.0, 'status': 'Unknown', 'recommendations': []}
        return {
            'er_predictions_24h': [],
            'er_times_24h': [],
            'er_mean_24h': 0.0,
            'icu_predictions_24h': [],
            'icu_peak': 0,
            'icu_peak_pct': 0.0,
            'staff_info': default_staff,
            'alerts': [],
            'recommendations': [],
            'icu_capacity': 0,
            'historical_mean_er': 0.0,
            # Bed defaults so templates referencing bed_latest always work
            'bed_latest': {'total_beds': 0, 'occupied': 0, 'available': 0, 'occupancy_pct': 0.0},
            'predicted_bed_demand': [],
            'peak_predicted_beds': 0,
            'peak_predicted_pct': 0.0,
            'message': str(e),
        }

    # ER admissions predictions (next 24 hours)
    er_preds = predict_next_24_hours(DATA_CSV)

    # Build corresponding next-24-hour timestamps starting from last historical timestamp +1h
    last_dt = df['datetime'].iloc[-1]
    er_times = [(last_dt + pd.Timedelta(hours=i+1)).strftime('%Y-%m-%d %H:%M') for i in range(len(er_preds))]

    # ICU predictions based on ER predictions
    icu_preds = predict_icu_demand(DATA_CSV, er_preds)

    # Staff workload and recommendations
    staff_info = compute_load_and_recommendations(er_preds, df)

    # Simple ICU capacity estimate: 15% of median total_beds
    median_total_beds = float(df['total_beds'].median())
    icu_capacity = median_total_beds * 0.15

    # Alerts
    alerts = []
    max_icu_pred = max(icu_preds)
    icu_usage_pct = (max_icu_pred / max(1.0, icu_capacity)) * 100.0
    if icu_usage_pct > 85.0:
        alerts.append({'type': 'ICU Overload', 'message': f'Predicted ICU usage {icu_usage_pct:.1f}% > 85%'})

    if staff_info['status'] == 'Critical':
        alerts.append({'type': 'Staff Burnout', 'message': 'Predicted staff workload is Critical'})

    historical_mean = float(df['er_admissions'].mean())
    historical_std = float(df['er_admissions'].std())
    if er_preds[0] > historical_mean + 2 * historical_std:
        alerts.append({'type': 'ER Surge', 'message': 'Next-hour ER admissions unusually high'})

    # Recommendations
    recommendations = []
    if max_icu_pred > icu_capacity:
        recommendations.append('Convert general beds to ICU where possible')
        recommendations.append('Call standby ICU staff')

    if staff_info['status'] in ['High', 'Critical']:
        recommendations.append('Add extra nursing shifts')

    if any(a['type'] == 'ER Surge' for a in alerts):
        recommendations.append('Activate emergency overflow ward')

    # Bed occupancy calculations (current/latest)
    bed_latest = {}
    try:
        latest = df.iloc[-1]
        total_beds_latest = int(latest.get('total_beds', 0))
        # prefer total_patients if available, else sum components
        if 'total_patients' in df.columns:
            occupied_latest = int(latest.get('total_patients', 0))
        else:
            icu_latest = int(latest.get('icu_beds_used', 0))
            normal_latest = int(latest.get('normal_beds_used', 0)) if 'normal_beds_used' in df.columns else 0
            emergency_latest = int(latest.get('emergency_patients', 0)) if 'emergency_patients' in df.columns else 0
            occupied_latest = icu_latest + normal_latest + emergency_latest

        available_latest = max(0, total_beds_latest - occupied_latest)
        occupancy_pct_latest = round((occupied_latest / total_beds_latest) * 100.0, 1) if total_beds_latest > 0 else 0.0
        bed_latest = {
            'total_beds': total_beds_latest,
            'occupied': occupied_latest,
            'available': available_latest,
            'occupancy_pct': occupancy_pct_latest,
        }
    except Exception:
        bed_latest = {'total_beds': 0, 'occupied': 0, 'available': 0, 'occupancy_pct': 0.0}

    # Store historical bed usage (append)
    try:
        import csv as _csv, datetime as _dt
        row = {
            'timestamp': _dt.datetime.now().isoformat(),
            'total_beds': bed_latest['total_beds'],
            'occupied': bed_latest['occupied'],
            'available': bed_latest['available'],
            'occupancy_pct': bed_latest['occupancy_pct'],
        }
        write_header = not os.path.exists(BED_HISTORY_CSV)
        with open(BED_HISTORY_CSV, 'a', newline='') as fh:
            writer = _csv.DictWriter(fh, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception:
        # non-fatal
        pass

    # Predict simple bed demand for next 24h: ICU preds + median normal beds used
    median_normal = int(df['normal_beds_used'].median()) if 'normal_beds_used' in df.columns else 0
    predicted_bed_demand = [int(round(ip + median_normal)) for ip in icu_preds]
    peak_predicted_beds = max(predicted_bed_demand) if predicted_bed_demand else 0
    peak_predicted_pct = round((peak_predicted_beds / max(1, median_total_beds)) * 100.0, 1) if median_total_beds > 0 else 0.0

    # Extra metrics
    er_mean_24 = float(np.mean(er_preds))
    peak_icu = int(round(max(icu_preds)))
    icu_peak_pct = (peak_icu / max(1.0, icu_capacity)) * 100.0

    # Merge and deduplicate recommendations (preserve order)
    merged_recs = []
    for r in recommendations + staff_info.get('recommendations', []):
        if r not in merged_recs:
            merged_recs.append(r)

    dashboard = {
        'er_predictions_24h': [round(float(x), 1) for x in er_preds],
        'er_times_24h': er_times,
        'er_mean_24h': round(er_mean_24, 1),
        'icu_predictions_24h': [int(round(x)) for x in icu_preds],
        'icu_peak': peak_icu,
        'icu_peak_pct': round(icu_peak_pct, 1),
        'staff_info': staff_info,
        'alerts': alerts,
        'recommendations': merged_recs,
        'icu_capacity': int(round(icu_capacity)),
        'historical_mean_er': round(historical_mean, 2),
        # Bed info (so templates can always access these keys)
        'bed_latest': bed_latest,
        'predicted_bed_demand': predicted_bed_demand,
        'peak_predicted_beds': peak_predicted_beds,
        'peak_predicted_pct': peak_predicted_pct,
    }

    return dashboard


@app.route('/upload', methods=['POST'])
def upload_csv():
    """Accept a CSV upload and replace the dataset, then retrain models."""
    if 'csv' not in request.files:
        return redirect(url_for('index', msg='No file part in request'))
    file = request.files['csv']
    if file.filename == '':
        return redirect(url_for('index', msg='No file selected'))
    filename = secure_filename(str(file.filename))
    if filename and filename.rsplit('.', 1)[-1].lower() in ALLOWED_EXT:
        import tempfile
        temp_path = DATA_CSV + '.upload.tmp'
        try:
            file.save(temp_path)
        except Exception as e:
            return redirect(url_for('index', msg=f'Error saving uploaded file: {e}'))

        ok, err = validate_csv_file(temp_path)
        if not ok:
            try:
                os.remove(temp_path)
            except Exception:
                pass
            return redirect(url_for('index', msg=f'CSV validation failed: {err}'))

        # Replace the live dataset atomically
        try:
            os.replace(temp_path, DATA_CSV)
        except Exception:
            # fallback to copy then remove
            try:
                import shutil
                shutil.copyfile(temp_path, DATA_CSV)
                os.remove(temp_path)
            except Exception as e:
                return redirect(url_for('index', msg=f'Failed to install uploaded CSV: {e}'))

        # retrain persisted models (best-effort)
        try:
            from model.er_prediction import predict_next_24_hours as _er
            from model.icu_prediction import predict_icu_demand as _icu
            _er(DATA_CSV, force_retrain=True)
            # call icu with a tiny demo list to trigger training
            _icu(DATA_CSV, [0, 0, 0], force_retrain=True)
        except Exception:
            # non-fatal: we still replaced the CSV but couldn't retrain
            return redirect(url_for('index', msg='CSV uploaded but retraining failed (check server logs)'))

        return redirect(url_for('index', msg='CSV uploaded and models retrained'))
    else:
        return redirect(url_for('index', msg='Invalid file type; please upload a .csv'))


@app.route('/retrain', methods=['POST'])
def retrain_models():
    """Force retrain persisted models from the current CSV."""
    try:
        from model.er_prediction import predict_next_24_hours as _er
        from model.icu_prediction import predict_icu_demand as _icu
        _er(DATA_CSV, force_retrain=True)
        _icu(DATA_CSV, [0, 0, 0], force_retrain=True)
        return redirect(url_for('index', msg='Models retrained successfully'))
    except Exception as e:
        return redirect(url_for('index', msg=f'Error retraining models: {e}'))


@app.route('/')
def index():
    # Allow optional message via query param (we avoid using flash to not require session)
    msg = request.args.get('msg')
    data = generate_dashboard_data()
    if msg:
        data['message'] = msg
    return render_template('index.html', data=data)


@app.route('/favicon.ico')
def favicon():
    # Return no-content to avoid 404 noise in server logs during demo
    return make_response(('', 204))


if __name__ == '__main__':
    # Run in debug mode for local development
    # Note: ensure you installed requirements.txt in the active interpreter
    app.run(host='127.0.0.1', port=5000, debug=True)

