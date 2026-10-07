# Heart Disease Prediction System - Architecture & Design

## 🏗️ System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Client Layer                                │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │  Web Browser     │  │   Mobile App     │  │  API Consumer  │ │
│  │ (Interactive UI) │  │  (Native/Web)    │  │  (Integration) │ │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬───────┘ │
└───────────┼──────────────────────┼──────────────────────┼────────┘
            │                      │                      │
            └──────────────────────┼──────────────────────┘
                                   │ HTTPS/REST
            ┌──────────────────────▼──────────────────────┐
            │         API Gateway Layer (Load Balancer)    │
            │  • Rate Limiting (100 req/min per IP)       │
            │  • Request Validation                        │
            │  • CORS Security Headers                     │
            │  • Request Logging & Monitoring              │
            └──────────────────────┬──────────────────────┘
                                   │
    ┌──────────────────────────────▼──────────────────────────────┐
    │            Application Layer (Flask)                         │
    │  ┌────────────────────────────────────────────────────────┐ │
    │  │  Prediction Service                                    │ │
    │  │  • Single Prediction (/api/predict)                    │ │
    │  │  • Batch Prediction (/api/predict/batch)              │ │
    │  │  • Model Metadata (/api/model/info)                    │ │
    │  │  • Health Check (/api/health)                          │ │
    │  └────────────────────────────────────────────────────────┘ │
    │  ┌────────────────────────────────────────────────────────┐ │
    │  │  Request Processing Pipeline                           │ │
    │  │  1. Input Validation (Schema + Value Checks)          │ │
    │  │  2. Feature Engineering (Encoding/Scaling)             │ │
    │  │  3. Model Inference                                    │ │
    │  │  4. Post-processing (Probability → Risk Category)      │ │
    │  │  5. Response Formatting & Metadata                     │ │
    │  └────────────────────────────────────────────────────────┘ │
    └───────────────┬──────────────────────────────────┬──────────┘
                    │                                  │
        ┌───────────▼─────────────┐      ┌────────────▼────────────┐
        │   ML Model Layer        │      │   Caching Layer        │
        │  ┌───────────────────┐  │      │  (Redis)               │
        │  │ Gradient Boosting │  │      │  • Model Cache         │
        │  │ Classifier        │  │      │  • Prediction Cache    │
        │  │                   │  │      │  • Feature Cache       │
        │  │ Accuracy: 87.5%   │  │      │  • TTL: 3600s          │
        │  │ ROC-AUC: 0.925    │  │      │                        │
        │  └───────────────────┘  │      └────────────────────────┘
        │  ┌───────────────────┐  │
        │  │ Feature Scaler    │  │
        │  │ (StandardScaler)  │  │
        │  └───────────────────┘  │
        │  ┌───────────────────┐  │
        │  │ Encoder/Decoder   │  │
        │  │ (Label Encoding)  │  │
        │  └───────────────────┘  │
        └───────────────────────┘
```

## 🔄 Request Processing Pipeline

### 1. **Single Prediction Flow**

```
POST /api/predict
├── Input Validation
│   ├── Schema validation (all 11 features present)
│   ├── Type checking (int/float)
│   └── Range validation (age: 20-80, BP: 60-300, etc.)
├── Cache Check
│   └── Return cached result if available
├── Feature Preprocessing
│   ├── Label encoding (categorical features)
│   ├── StandardScaler normalization
│   └── Feature vector assembly
├── Model Inference
│   └── Gradient Boosting prediction + probability
├── Post-processing
│   ├── Sigmoid → probability conversion
│   └── Risk category assignment (low/moderate/high)
├── Cache Store
│   └── Store result with TTL
└── Response
    └── Return JSON with prediction, probability, confidence
```

### 2. **Batch Prediction Flow**

```
POST /api/predict/batch
├── Batch Validation
│   ├── Size check (max 1000 records)
│   ├── Individual input validation
│   └── Error aggregation
├── Parallel Processing
│   └── Process up to 100 records concurrently
├── Results Aggregation
│   └── Combine all predictions
└── Response
    └── Return batch_id, predictions[], statistics
```

## 💾 Data Models

### Patient Input Schema
```json
{
  "age": 45,                          # int, 20-120
  "sex": 1,                            # int, 0=Female|1=Male
  "chest_pain_type": 1,                # int, 0-3
  "resting_bp": 130,                   # int, 60-300 mmHg
  "cholesterol": 240,                  # int, 0-500 mg/dL
  "fasting_bs": 0,                     # int, 0|1 (≤120 or >120 mg/dL)
  "resting_ecg": 0,                    # int, 0|1|2
  "max_hr": 150,                       # int, 50-250
  "exercise_angina": 0,                # int, 0|1
  "oldpeak": 1.5,                      # float, 0-6.2 (ST depression)
  "st_slope": 1                        # int, 0|1|2
}
```

### Prediction Response Schema
```json
{
  "prediction": 0,                     # Binary: 0=No disease, 1=Disease
  "probability": 22.5,                 # Percentage: 0-100
  "risk_level": "low",                 # Category: low|moderate|high
  "confidence": 95.2,                  # Model confidence: 0-100
  "timestamp": "2024-06-15T10:30:45Z", # ISO 8601
  "model_version": "2.0.0"             # Model version
}
```

## 🤖 Machine Learning Pipeline

### Training Phase
```
1. Data Loading (heart.csv)
   └── 918 samples, 11 features, 1 target

2. Data Preprocessing
   ├── Handle missing values (none found)
   ├── Encode categorical features (Label Encoding)
   ├── Feature scaling (StandardScaler)
   └── Train-test split (80-20, stratified)

3. Model Development
   ├── Baseline: Logistic Regression
   ├── Ensemble: Random Forest
   └── Best: Gradient Boosting (87.5% accuracy)

4. Hyperparameter Tuning
   ├── GridSearchCV with StratifiedKFold (5-fold)
   ├── Scoring metric: ROC-AUC
   └── Parameter ranges: n_estimators, learning_rate, max_depth

5. Model Evaluation
   ├── Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
   ├── Cross-validation results
   ├── Feature importance analysis
   └── Model persistence (pickle format)
```

### Inference Phase
```
1. Input Validation
   └── Check types, ranges, required fields

2. Feature Preprocessing
   ├── Apply same scaling (StandardScaler)
   ├── Apply same encoding
   └── Ensure feature order matches training

3. Model Prediction
   ├── Get probability from model
   ├── Convert to percentage (0-100)
   └── Calculate confidence score

4. Post-processing
   ├── Assign risk category
   ├── Add model metadata
   └── Format response
```

## 🔒 Security Architecture

### Authentication & Authorization
- **Current**: No authentication (MVP stage)
- **Production**: API Key or OAuth 2.0
- **Rate Limiting**: 100 requests/minute per IP address
- **Input Validation**: All inputs validated against schema + value ranges

### Data Security
- **HTTPS Only**: Enforce TLS 1.2+
- **CORS Headers**: Restrict to authorized origins
- **Input Sanitization**: Strip special characters, validate types
- **No PII Logging**: Exclude sensitive patient data from logs

### Infrastructure Security
- **Container**: Non-root user (UID 1000)
- **Health Checks**: Automatic restart on failure
- **Environment Variables**: Sensitive data in .env files
- **Secrets Management**: Use HashiCorp Vault (production)

## 📊 Model Performance

| Metric | Value | Benchmark |
|--------|-------|-----------|
| Accuracy | 87.5% | ≥80% |
| Precision | 85.2% | ≥80% |
| Recall | 89.1% | ≥85% |
| F1-Score | 87.2% | ≥85% |
| ROC-AUC | 0.925 | ≥0.90 |

### Model Comparison
```
Model                 | Accuracy | ROC-AUC
─────────────────────┼──────────┼────────
Logistic Regression  |   84.5%  | 0.908
Random Forest        |   86.2%  | 0.920
Gradient Boosting ⭐ |   87.5%  | 0.925
```

### Feature Importance (Gradient Boosting)
```
ST_Slope           ████████████████████ 24.5%
ExerciseAngina     █████████████████░░░ 19.8%
Oldpeak            ████████████████░░░░ 17.6%
MaxHR              ██████████░░░░░░░░░░ 13.4%
Age                ██████░░░░░░░░░░░░░░  8.9%
Sex                █████░░░░░░░░░░░░░░░  7.8%
RestingBP          ███░░░░░░░░░░░░░░░░░  4.5%
Cholesterol        ██░░░░░░░░░░░░░░░░░░  3.5%
```

## 🚀 Deployment Architecture

### Local Development (Docker Compose)
```
docker-compose.yml
├── Flask API Service (Port 5000)
│   └── Health check: /api/health
├── Redis Cache Service (Port 6379)
│   └── Model & prediction caching
└── Shared Network: heart-network
```

### Production Deployment Options

#### Option 1: Container Registry + Cloud Run / App Service
```
Docker Image Registry (ACR/ECR)
    ↓
Azure Container Instances / AWS ECS
    ↓
Load Balancer
    ↓
Multiple API Replicas
    ↓
Managed Cache (Redis/Memcached)
    ↓
Monitoring (Application Insights / CloudWatch)
```

#### Option 2: Kubernetes (AKS / EKS)
```
Helm Chart (heart-prediction)
├── Deployment
│   ├── 3-5 replicas
│   ├── Resource limits
│   └── Health probes
├── Service
│   └── Load balancer
├── Ingress
│   └── HTTPS termination
└── ConfigMap
    └── Model configuration
```

#### Option 3: Serverless (Azure Functions / AWS Lambda)
```
Azure Functions
├── HTTP Trigger: /api/predict
├── Scale: 0-100+ instances
├── Cold start: ~2 seconds
├── Cost: Pay per execution
└── Storage: Azure Blob (model)
```

## 📈 Monitoring & Observability

### Metrics
```
Application Metrics:
├── Predictions/sec (throughput)
├── Average response time (p50, p95, p99)
├── Error rate (4xx, 5xx)
├── Cache hit ratio
└── Model inference latency

Business Metrics:
├── Risk distribution (low/moderate/high)
├── False positive rate
├── False negative rate
└── Model drift detection
```

### Logging
```
Log Levels:
├── DEBUG: Feature preprocessing, model inference details
├── INFO: Prediction requests, cache hits, health status
├── WARNING: Input validation failures, slow requests
└── ERROR: Model failures, system errors

Log Format:
{
  "timestamp": "2024-06-15T10:30:45Z",
  "level": "INFO",
  "service": "prediction-api",
  "request_id": "uuid",
  "message": "Prediction completed",
  "duration_ms": 45,
  "model_version": "2.0.0"
}
```

### Alerts
```
Critical:
├── API error rate > 5%
├── Response time (p95) > 1s
├── Model inference failure
└── Health check failures

Warning:
├── Cache hit rate < 50%
├── Prediction latency > 500ms
└── Memory usage > 80%
```

## 🔄 CI/CD Pipeline

```
GitHub Push
    ↓
GitHub Actions Workflow
├── Lint (pylint, black)
├── Unit Tests (pytest)
├── Integration Tests
├── Docker Build
├── Security Scan (Trivy)
├── Push to Registry
└── Deploy to Staging
    ↓
Manual Approval
    ↓
Deploy to Production
├── Blue-green deployment
├── Health checks
├── Smoke tests
└── Rollback on failure
```

## 🔧 Configuration Management

### Environment Variables
```bash
FLASK_ENV=production|development
DEBUG=False
LOG_LEVEL=INFO
MODEL_PATH=./models/model_advanced.pkl
SCALER_PATH=./models/scaler.pkl
REDIS_URL=redis://localhost:6379
BATCH_SIZE_LIMIT=1000
RATE_LIMIT=100/minute
CACHE_TTL=3600
```

### Feature Flags
```python
FEATURES = {
    'batch_prediction': True,
    'caching_enabled': True,
    'model_versioning': True,
    'feature_importance': True,
    'explainability': False,  # SHAP (future)
}
```

## 📋 Design Decisions & Trade-offs

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| Gradient Boosting | Best accuracy (87.5%) | Slightly slower inference than LR |
| StandardScaler | Handles outliers well | Requires scaling at inference |
| Label Encoding | Simple, efficient | Not as interpretable as one-hot |
| Redis Caching | Fast prediction lookup | Requires cache invalidation logic |
| Flask (not FastAPI) | Simplicity, learning curve | Lower async performance |
| Docker | Reproducible deployment | Slight overhead vs native |
| OpenAPI/Swagger | API documentation | Extra maintenance |

## 🎓 Educational Value for Interview

This architecture demonstrates:
1. **Full ML Pipeline**: Data → Training → Evaluation → Deployment
2. **Production Concerns**: Logging, monitoring, security, caching
3. **API Design**: REST principles, validation, error handling
4. **Infrastructure**: Docker, containerization, cloud deployment
5. **Testing**: Unit, integration, edge cases, performance
6. **Documentation**: Architecture, API specs, design decisions
7. **Best Practices**: SOLID principles, separation of concerns, scalability

