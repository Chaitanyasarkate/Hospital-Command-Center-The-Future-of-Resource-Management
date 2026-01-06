# AI-Based Predictive Hospital Resource & Emergency Load Intelligence System

Run locally (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000/> to view the dashboard.

Project structure:

- `app.py` - Flask backend that runs models and renders dashboard
- `data/hospital_data.csv` - synthetic realistic hourly hospital data
- `model/er_prediction.py` - ER admissions model (LinearRegression)
- `model/icu_prediction.py` - ICU demand model (LinearRegression)
- `model/staff_load.py` - staff workload and recommendations
- `templates/index.html` - dashboard UI
- `static/*` - CSS and JS

Notes:
- Models are intentionally simple and explainable for hackathon/demo purposes.
- For production, persist trained models, add validation, and secure the web app.

