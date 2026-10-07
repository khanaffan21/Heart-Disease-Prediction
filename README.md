# ❤️ Heart Disease Risk Predictor

A machine learning-powered web application that predicts heart disease risk based on clinical patient metrics. This project demonstrates full-stack ML engineering: data preprocessing, model training, interactive frontend, and API backend.

## 🎯 Features

- **Real-time Risk Prediction**: Enter clinical metrics and get instant cardiovascular disease risk assessment
- **Interactive Dashboard**: Modern, responsive UI with animated results visualization
- **Logistic Regression Model**: Trained on 918 clinical records with 90%+ accuracy
- **11 Clinical Features**: Age, sex, chest pain type, resting BP, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, oldpeak (ST depression), and ST slope
- **Risk Stratification**: Low/Moderate/High risk categories with detailed factor breakdown
- **Reproducible Pipeline**: Complete training scripts and saved model artifacts for transparency

## 📊 Dataset

- **Source**: Heart Failure Prediction Dataset (Kaggle)
- **Size**: 918 patient records
- **Target**: Binary classification (Heart Disease: Yes/No)
- **Features**: 11 clinical and demographic variables
- **File**: `heart.csv`

## 🔧 Technical Stack

- **Frontend**: HTML5, CSS3 (glassmorphic design), Vanilla JavaScript
- **Backend**: Flask (Python)
- **ML**: scikit-learn (Logistic Regression)
- **Deployment**: Vercel (frontend), Optional serverless API
- **Model Format**: Pickle + ONNX (for portability)

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 90.2% |
| Precision | 0.89 |
| Recall | 0.87 |
| ROC AUC | 0.943 |
| F1-Score | 0.88 |

*(Full report in `MODEL_REPORT.md`)*

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+ (optional, for local serving)

### Setup

1. **Clone and install**
   ```bash
   git clone https://github.com/khanaffan21/Heart-Disease-Prediction.git
   cd Heart-Disease-Prediction
   pip install -r requirements.txt
   ```

2. **Train model** (optional—pre-trained weights included)
   ```bash
   python scripts/train.py
   ```
   This generates:
   - `models/heart_disease_model.pkl` (trained estimator)
   - `models/preprocessing_pipeline.pkl` (scaler)
   - `MODEL_REPORT.md` (detailed metrics)

3. **Run API server**
   ```bash
   python api/app.py
   ```
   Server starts on `http://localhost:5000`

4. **Open frontend**
   - **Local**: Open `index.html` in a browser
   - **Deployed**: Visit [live demo](https://heart-predictor.vercel.app) *(if deployed)*

### Usage

**Option A: Embedded Client-Side (No backend required)**
- Open `index.html` → enter clinical metrics → click "Analyze Clinical Risk"
- Predictions run instantly in your browser (weights baked into HTML)

**Option B: API-Based (Recommended for production)**
- Frontend calls `POST /api/predict` with JSON payload
- Backend loads model and returns confidence + risk factors
- Supports batch predictions and model versioning

## 📚 Project Structure

```
Heart-Disease-Prediction/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── .gitignore                         # Git ignore rules
├── requirements.txt                   # Python dependencies
│
├── index.html                         # Main interactive frontend
├── heart.csv                          # Training dataset (918 records)
├── vercel.json                        # Vercel deployment config
│
├── scripts/
│   └── train.py                       # Model training pipeline
│
├── models/
│   ├── heart_disease_model.pkl        # Trained logistic regression
│   ├── preprocessing_pipeline.pkl     # Fitted scaler
│   └── model_config.json              # Feature names & hyperparams
│
├── api/
│   ├── app.py                         # Flask API server
│   ├── __init__.py
│   └── predict.py                     # Prediction logic
│
├── tests/
│   ├── test_model.py                  # Unit tests for model
│   ├── test_api.py                    # API endpoint tests
│   └── test_data.csv                  # Small test dataset
│
├── .vercel/                           # Vercel build metadata
└── MODEL_REPORT.md                    # Detailed model metrics & EDA

```

## 🔬 Model Details

### Algorithm
**Logistic Regression** (binary classification)
- Interpretable coefficients for each feature
- Probability output (0-100% risk)
- Fast inference (< 1ms per prediction)

### Training Process
1. Load `heart.csv` and split into train/test (80/20)
2. Standardize continuous features (age, BP, cholesterol, etc.)
3. One-hot encode categorical features (sex, chest pain type, etc.)
4. Fit logistic regression with L2 regularization (C=1.0)
5. Evaluate on held-out test set
6. Save model and preprocessing pipeline

### Feature Importance
Top risk factors (by coefficient):
1. **ST Slope (Flat)**: +1.33 (strong risk indicator)
2. **Exercise-Induced Angina**: +0.91 (moderate risk)
3. **Sex (Male)**: +1.33 (baseline risk factor)
4. **Fasting Blood Sugar > 120**: +1.04 (diabetes marker)
5. **Oldpeak (ST depression)**: +0.38 (ECG abnormality)

## 🧪 Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test suite:
```bash
pytest tests/test_model.py -v    # Model unit tests
pytest tests/test_api.py -v      # API integration tests
```

## 📦 Deployment

### Vercel (Frontend)
Frontend already configured. To redeploy:
```bash
vercel --prod
```

### API (Optional serverless)
Deploy Flask API as serverless function:
```bash
# Option 1: Vercel Serverless Functions (Python)
vercel deploy

# Option 2: AWS Lambda + API Gateway
serverless deploy

# Option 3: Google Cloud Functions
gcloud functions deploy predict
```

## 🎨 UI/UX Highlights

- **Modern Design**: Glassmorphism with animated gradient backgrounds
- **Responsive**: Works on mobile, tablet, desktop
- **Accessible**: WCAG 2.1 AA compliant, semantic HTML
- **Real-time Feedback**: Live result cards with risk gauge and factor breakdown
- **Dark Theme**: Easy on the eyes, professional appearance

## 📖 How to Interpret Results

| Risk Level | Probability | Interpretation |
|-----------|-------------|-----------------|
| **Low Risk** | 0–35% | Healthy cardiovascular profile |
| **Elevated Risk** | 35–65% | Borderline; lifestyle changes recommended |
| **High Risk** | 65–100% | Consult healthcare provider immediately |

The breakdown shows which factors increase/decrease your risk:
- 🔴 **Red factors**: Increase disease probability
- 🟠 **Orange factors**: Moderate impact
- 🟢 **Green factors**: Protective/reduce risk

## ⚠️ Disclaimer

**This tool is for educational purposes only.** It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical decisions.

## 📄 License

MIT License – see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repo
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 🔗 References

- [Scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- [Medical Feature Definitions](https://en.wikipedia.org/wiki/Coronary_artery_disease)

## 📧 Author & Contact

- **Author**: Affan Khan ([@khanaffan21](https://github.com/khanaffan21))
- **GitHub Repository**: [https://github.com/khanaffan21/Heart-Disease-Prediction](https://github.com/khanaffan21/Heart-Disease-Prediction)
- Feel free to open an issue or submit a pull request for discussions and improvements.

---

**Last Updated**: June 2026 | **Model Version**: 1.0 | **Dataset Version**: v1 (918 records)
