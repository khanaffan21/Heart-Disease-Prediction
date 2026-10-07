"""
Flask API for Heart Disease Prediction

Provides REST endpoints for:
- POST /api/predict - Single prediction
- POST /api/predict/batch - Batch predictions
- GET /api/model/info - Model metadata
- GET /health - Health check
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import json
import os
import numpy as np
from datetime import datetime
import sys

app = Flask(__name__)
CORS(app)

# Configuration
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'heart_disease_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'preprocessing_pipeline.pkl')
CONFIG_PATH = os.path.join(MODELS_DIR, 'model_config.json')

# Global model storage
model = None
scaler = None
config = None


def load_model_artifacts():
    """Load model, scaler, and config from disk."""
    global model, scaler, config
    
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        
        print(f"✓ Model loaded successfully")
        print(f"  - Features: {len(config['feature_names'])}")
        print(f"  - Model type: {config['model_type']}")
        return True
    
    except FileNotFoundError as e:
        print(f"⚠️  Model files not found. Please run: python scripts/train.py")
        print(f"   Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    }), 200


@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Return model metadata."""
    if not config:
        return jsonify({'error': 'Model not loaded'}), 503
    
    return jsonify({
        'model_type': config['model_type'],
        'n_features': config['n_features'],
        'feature_names': config['feature_names'],
        'hyperparameters': config['hyperparameters'],
        'training_date': config['training_date']
    }), 200


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Single prediction endpoint.
    
    Expected JSON:
    {
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
    }
    """
    if not model or not config:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate required features
        missing = [f for f in config['feature_names'] if f not in data]
        if missing:
            return jsonify({
                'error': f'Missing features: {missing}'
            }), 400
        
        # Build feature vector in correct order
        features = np.array([
            [data[fname] for fname in config['feature_names']]
        ])
        
        # Predict
        probability = model.predict_proba(features)[0][1]
        prediction = int(model.predict(features)[0])
        risk_pct = float(probability * 100)
        
        # Determine risk level
        if probability < 0.35:
            risk_level = "Low"
            risk_color = "green"
        elif probability < 0.65:
            risk_level = "Moderate"
            risk_color = "orange"
        else:
            risk_level = "High"
            risk_color = "red"
        
        return jsonify({
            'prediction': prediction,
            'probability': float(probability),
            'risk_percentage': risk_pct,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except KeyError as e:
        return jsonify({'error': f'Invalid feature name: {e}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint.
    
    Expected JSON:
    {
        "records": [
            {...patient1...},
            {...patient2...}
        ]
    }
    """
    if not model or not config:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        data = request.get_json()
        
        if not data or 'records' not in data:
            return jsonify({'error': 'Expected {"records": [...]}'})
        
        records = data['records']
        if not isinstance(records, list):
            return jsonify({'error': 'records must be a list'}), 400
        
        results = []
        for i, record in enumerate(records):
            try:
                # Build feature vector
                features = np.array([
                    [record[fname] for fname in config['feature_names']]
                ])
                
                prob = float(model.predict_proba(features)[0][1])
                pred = int(model.predict(features)[0])
                risk_pct = prob * 100
                
                if prob < 0.35:
                    risk_level = "Low"
                elif prob < 0.65:
                    risk_level = "Moderate"
                else:
                    risk_level = "High"
                
                results.append({
                    'record_index': i,
                    'prediction': pred,
                    'probability': prob,
                    'risk_percentage': risk_pct,
                    'risk_level': risk_level
                })
            
            except Exception as e:
                results.append({
                    'record_index': i,
                    'error': str(e)
                })
        
        return jsonify({
            'total_records': len(records),
            'successful': len([r for r in results if 'error' not in r]),
            'results': results
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500


@app.route('/api/predict/explain', methods=['POST'])
def predict_explain():
    """
    Prediction with feature importance explanation.
    """
    if not model or not config:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        data = request.get_json()
        
        # Build feature vector
        features = np.array([
            [data[fname] for fname in config['feature_names']]
        ])
        
        probability = float(model.predict_proba(features)[0][1])
        risk_pct = probability * 100
        
        # Get feature contributions (coef * feature_value)
        contributions = model.coef_[0] * features[0]
        
        # Sort by absolute contribution
        feature_contrib = list(zip(
            config['feature_names'],
            contributions,
            features[0]
        ))
        feature_contrib.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Top contributing factors
        top_factors = []
        for fname, contrib, value in feature_contrib[:5]:
            direction = "increases" if contrib > 0 else "decreases"
            top_factors.append({
                'feature': fname,
                'value': float(value),
                'coefficient': float(contrib),
                'contribution': float(contrib),
                'direction': direction
            })
        
        return jsonify({
            'probability': probability,
            'risk_percentage': risk_pct,
            'top_factors': top_factors,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': [
            'GET /health',
            'GET /api/model/info',
            'POST /api/predict',
            'POST /api/predict/batch',
            'POST /api/predict/explain'
        ]
    }), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("❤️  HEART DISEASE PREDICTION API")
    print("=" * 60)
    
    # Load model
    if load_model_artifacts():
        print("\n🚀 Starting API server...")
        print(f"📍 http://localhost:5000")
        print(f"📚 Docs: http://localhost:5000/api/model/info")
        print(f"💊 Test prediction: curl -X POST http://localhost:5000/api/predict")
        print("\nPress Ctrl+C to stop\n")
        
        app.run(debug=True, port=5000)
    else:
        print("\n❌ Cannot start API without model. Run training first:")
        print("   python scripts/train.py")
        sys.exit(1)
