# Quick Start Guide

Get the Heart Disease Predictor running locally in 5 minutes.

## Prerequisites

- **Python 3.8+** ([download](https://www.python.org/downloads/))
- **Git** (optional, for cloning)

## Option A: Frontend Only (No Backend Required)

Perfect for quick testing—predictions run instantly in your browser.

```bash
# 1. Navigate to project folder
cd Heart-Disease-Prediction

# 2. Open in browser
# On Windows (PowerShell):
Start-Process index.html

# On macOS:
open index.html

# On Linux:
xdg-open index.html
```

✅ Done! Enter clinical metrics and click "Analyze Clinical Risk" to see predictions.

---

## Option B: Full Setup with Reproducible Model & API

Train the model from scratch and run a Flask API backend.

### Step 1: Install Dependencies

```bash
# Navigate to project
cd Heart-Disease-Prediction

# Create virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### Step 2: Train the Model

```bash
python scripts/train.py
```

**Output:**
```
❤️  HEART DISEASE PREDICTION MODEL TRAINING
============================================================
✓ Directories ready

📊 Dataset loaded:
   Shape: 918 samples × 12 features
   ...

⚙️ Preprocessing data...
🚀 Training model...
📈 Evaluating model...
   Accuracy:  0.9022
   Precision: 0.8945
   Recall:    0.8723
   ...

💾 Saving artifacts...
   ✓ Model saved to models/heart_disease_model.pkl
   ✓ Scaler saved to models/preprocessing_pipeline.pkl
   ✓ Config saved to models/model_config.json

📄 Generating report...
   ✓ Report saved to MODEL_REPORT.md

✅ Training pipeline complete!
```

**Generated Files:**
- `models/heart_disease_model.pkl` – Trained logistic regression
- `models/preprocessing_pipeline.pkl` – Feature scaler
- `models/model_config.json` – Feature names & hyperparameters
- `MODEL_REPORT.md` – Detailed metrics and analysis

### Step 3: Start API Server

```bash
python api/app.py
```

**Output:**
```
❤️  HEART DISEASE PREDICTION API
============================================================
✓ Model loaded successfully
  - Features: 15
  - Model type: LogisticRegression

🚀 Starting API server...
📍 http://localhost:5000
📚 Docs: http://localhost:5000/api/model/info
💊 Test prediction: curl -X POST http://localhost:5000/api/predict

Press Ctrl+C to stop
```

### Step 4: Test API (in another terminal)

```bash
# Get model info
curl http://localhost:5000/api/model/info

# Make a prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 50,
    "RestingBP": 130,
    "Cholesterol": 220,
    "FastingBS": 0,
    "MaxHR": 140,
    "Oldpeak": 1.0,
    "Sex_M": 1,
    "ChestPainType_ATA": 0,
    "ChestPainType_NAP": 0,
    "ChestPainType_TA": 0,
    "RestingECG_Normal": 1,
    "RestingECG_ST": 0,
    "ExerciseAngina_Y": 0,
    "ST_Slope_Flat": 1,
    "ST_Slope_Up": 0
  }'

# Expected response:
# {
#   "prediction": 1,
#   "probability": 0.72,
#   "risk_percentage": 72.0,
#   "risk_level": "High",
#   "risk_color": "red",
#   "timestamp": "2026-06-15T..."
# }
```

### Step 5: Open Frontend (Optional)

In a third terminal, open the frontend:

```bash
# Windows (PowerShell)
Start-Process index.html

# macOS
open index.html

# Linux
xdg-open index.html
```

The frontend can now optionally call the API for predictions (currently uses embedded weights).

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Model tests only
pytest tests/test_model.py -v

# API tests only
pytest tests/test_api.py -v

# With coverage
pytest tests/ --cov=scripts --cov=api --cov-report=html
```

---

## Project Structure

```
Heart-Disease-Prediction/
├── index.html                    # Main interactive frontend
├── heart.csv                     # Training dataset (918 records)
│
├── scripts/
│   └── train.py                  # Model training pipeline
│
├── models/                       # (Created after running train.py)
│   ├── heart_disease_model.pkl   # Trained model
│   ├── preprocessing_pipeline.pkl
│   └── model_config.json
│
├── api/
│   ├── app.py                    # Flask API server
│   └── __init__.py
│
├── tests/
│   ├── test_model.py             # Unit tests
│   └── test_api.py
│
├── README.md                     # Full documentation
├── MODEL_REPORT.md               # (Generated after train.py)
├── LICENSE                       # MIT License
├── requirements.txt              # Dependencies
└── .gitignore
```

---

## Common Issues

### Issue: `ModuleNotFoundError: No module named 'sklearn'`
**Solution:** Install dependencies: `pip install -r requirements.txt`

### Issue: `FileNotFoundError: models/heart_disease_model.pkl`
**Solution:** Train the model first: `python scripts/train.py`

### Issue: API server won't start
**Solution:** Check port 5000 is free or modify `app.py` to use a different port

### Issue: Flask CORS errors
**Solution:** Already included in requirements.txt; reinstall: `pip install flask-cors --upgrade`

---

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/health` | Server status check |
| `GET` | `/api/model/info` | Model metadata |
| `POST` | `/api/predict` | Single prediction |
| `POST` | `/api/predict/batch` | Multiple predictions |
| `POST` | `/api/predict/explain` | Prediction with feature importance |

---

## Next Steps

1. **Explore Model Report**: `cat MODEL_REPORT.md`
2. **Understand Features**: Read [README.md](README.md)
3. **Deploy**: Use Vercel (frontend) or serverless (API)
4. **Customize**: Modify model hyperparameters in `scripts/train.py`
5. **Integrate**: Call API from your own frontend

---

## Getting Help

- **Model questions?** → Check `MODEL_REPORT.md`
- **API documentation?** → Visit `http://localhost:5000/api/model/info`
- **Feature definitions?** → See [README.md](README.md#-model-details)
- **Issues?** → Review `.gitignore` and ensure all files are in place

---

**Happy predicting!** ❤️
