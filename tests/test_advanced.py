"""
Advanced Test Suite for Heart Disease Prediction System

Comprehensive testing including:
- Unit tests for ML pipeline
- Integration tests for API endpoints
- Edge case testing
- Performance testing
- Validation and error handling
- Data integrity tests
"""

import pytest
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.app import app, load_model, preprocess_input
from scripts.train import train_model, load_data, evaluate_model


@pytest.fixture
def client():
    """Flask test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def valid_input():
    """Valid patient input for testing."""
    return {
        "age": 45,
        "sex": 1,
        "chest_pain_type": 1,
        "resting_bp": 130,
        "cholesterol": 240,
        "fasting_bs": 0,
        "resting_ecg": 0,
        "max_hr": 150,
        "exercise_angina": 0,
        "oldpeak": 1.5,
        "st_slope": 1
    }


# ============================================================================
# UNIT TESTS - Data Preprocessing
# ============================================================================

class TestDataPreprocessing:
    """Test data loading and preprocessing."""
    
    def test_load_data(self):
        """Test data loading."""
        df = load_data()
        assert df is not None
        assert len(df) > 0
        assert 'HeartDisease' in df.columns
    
    def test_data_shape(self):
        """Test data dimensions."""
        df = load_data()
        assert df.shape[0] == 918  # Expected sample count
        assert df.shape[1] == 12   # 11 features + 1 target
    
    def test_no_missing_values(self):
        """Test data has no missing values."""
        df = load_data()
        assert df.isnull().sum().sum() == 0
    
    def test_target_distribution(self):
        """Test target variable distribution."""
        df = load_data()
        target_counts = df['HeartDisease'].value_counts()
        # Verify balanced dataset
        assert 0 in target_counts.index
        assert 1 in target_counts.index
        # Verify reasonable class balance (30-70%)
        ratio = target_counts[1] / len(df)
        assert 0.3 < ratio < 0.7
    
    def test_feature_ranges(self):
        """Test features are within expected ranges."""
        df = load_data()
        assert df['Age'].min() >= 20 and df['Age'].max() <= 80
        assert df['RestingBP'].min() >= 60 and df['RestingBP'].max() <= 300
        assert df['Cholesterol'].min() >= 0 and df['Cholesterol'].max() <= 500
        assert df['MaxHR'].min() >= 50 and df['MaxHR'].max() <= 250
    
    def test_categorical_values(self):
        """Test categorical features have expected values."""
        df = load_data()
        assert df['Sex'].isin([0, 1]).all()
        assert df['ChestPainType'].isin([0, 1, 2, 3]).all()
        assert df['FastingBS'].isin([0, 1]).all()
        assert df['RestingECG'].isin([0, 1, 2]).all()
        assert df['ExerciseAngina'].isin([0, 1]).all()
        assert df['ST_Slope'].isin([0, 1, 2]).all()


# ============================================================================
# UNIT TESTS - Model Training
# ============================================================================

class TestModelTraining:
    """Test model training and evaluation."""
    
    def test_train_model(self):
        """Test model training completes successfully."""
        model, metrics = train_model()
        assert model is not None
        assert metrics is not None
    
    def test_model_metrics(self):
        """Test model performance metrics."""
        model, metrics = train_model()
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics
    
    def test_metrics_ranges(self):
        """Test metrics are valid (0-1 range)."""
        model, metrics = train_model()
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            assert 0 <= metrics[metric] <= 1
    
    def test_minimum_accuracy(self):
        """Test model meets minimum accuracy threshold."""
        model, metrics = train_model()
        assert metrics['accuracy'] > 0.80  # At least 80% accuracy
    
    def test_model_reproducibility(self):
        """Test model training is reproducible."""
        model1, metrics1 = train_model(random_state=42)
        model2, metrics2 = train_model(random_state=42)
        # Check same random state produces same metrics
        assert metrics1['accuracy'] == metrics2['accuracy']


# ============================================================================
# UNIT TESTS - Input Validation
# ============================================================================

class TestInputValidation:
    """Test input validation and error handling."""
    
    def test_valid_input(self, valid_input):
        """Test valid input is accepted."""
        result = preprocess_input(valid_input)
        assert result is not None
        assert len(result) == 11  # 11 features
    
    def test_missing_field(self, valid_input):
        """Test missing required field raises error."""
        del valid_input['age']
        with pytest.raises(ValueError):
            preprocess_input(valid_input)
    
    def test_invalid_age(self, valid_input):
        """Test invalid age values."""
        valid_input['age'] = 150  # Too high
        with pytest.raises(ValueError):
            preprocess_input(valid_input)
        
        valid_input['age'] = 10  # Too low
        with pytest.raises(ValueError):
            preprocess_input(valid_input)
    
    def test_invalid_sex(self, valid_input):
        """Test invalid sex value."""
        valid_input['sex'] = 2
        with pytest.raises(ValueError):
            preprocess_input(valid_input)
    
    def test_invalid_bp(self, valid_input):
        """Test invalid blood pressure."""
        valid_input['resting_bp'] = 400  # Too high
        with pytest.raises(ValueError):
            preprocess_input(valid_input)
    
    def test_invalid_cholesterol(self, valid_input):
        """Test invalid cholesterol."""
        valid_input['cholesterol'] = 600  # Too high
        with pytest.raises(ValueError):
            preprocess_input(valid_input)
    
    def test_invalid_max_hr(self, valid_input):
        """Test invalid maximum heart rate."""
        valid_input['max_hr'] = 300  # Too high
        with pytest.raises(ValueError):
            preprocess_input(valid_input)
    
    def test_negative_oldpeak(self, valid_input):
        """Test negative ST depression."""
        valid_input['oldpeak'] = -1
        with pytest.raises(ValueError):
            preprocess_input(valid_input)
    
    def test_invalid_categorical(self, valid_input):
        """Test invalid categorical values."""
        valid_input['chest_pain_type'] = 5
        with pytest.raises(ValueError):
            preprocess_input(valid_input)


# ============================================================================
# INTEGRATION TESTS - API Endpoints
# ============================================================================

class TestAPIEndpoints:
    """Test Flask API endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get('/api/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] in ['healthy', 'degraded']
    
    def test_model_info(self, client):
        """Test model info endpoint."""
        response = client.get('/api/model/info')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'model_name' in data
        assert 'accuracy' in data
        assert 'precision' in data
    
    def test_single_prediction(self, client, valid_input):
        """Test single prediction endpoint."""
        response = client.post('/api/predict',
                              json=valid_input,
                              content_type='application/json')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'probability' in data
        assert 'risk_level' in data
        assert 0 <= data['probability'] <= 100
    
    def test_prediction_missing_field(self, client, valid_input):
        """Test prediction with missing field."""
        del valid_input['age']
        response = client.post('/api/predict',
                              json=valid_input,
                              content_type='application/json')
        assert response.status_code in [400, 422]
    
    def test_prediction_invalid_input(self, client, valid_input):
        """Test prediction with invalid input."""
        valid_input['age'] = 200
        response = client.post('/api/predict',
                              json=valid_input,
                              content_type='application/json')
        assert response.status_code in [400, 422]
    
    def test_batch_prediction(self, client):
        """Test batch prediction endpoint."""
        batch_data = {
            "patients": [
                {
                    "age": 45, "sex": 1, "chest_pain_type": 1, "resting_bp": 130,
                    "cholesterol": 240, "fasting_bs": 0, "resting_ecg": 0,
                    "max_hr": 150, "exercise_angina": 0, "oldpeak": 1.5, "st_slope": 1
                },
                {
                    "age": 55, "sex": 0, "chest_pain_type": 0, "resting_bp": 120,
                    "cholesterol": 200, "fasting_bs": 0, "resting_ecg": 0,
                    "max_hr": 160, "exercise_angina": 0, "oldpeak": 0, "st_slope": 0
                }
            ]
        }
        response = client.post('/api/predict/batch',
                              json=batch_data,
                              content_type='application/json')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['total_records'] == 2
        assert len(data['predictions']) == 2
    
    def test_batch_size_limit(self, client):
        """Test batch size limit enforcement."""
        large_batch = {
            "patients": [
                {
                    "age": 45, "sex": 1, "chest_pain_type": 1, "resting_bp": 130,
                    "cholesterol": 240, "fasting_bs": 0, "resting_ecg": 0,
                    "max_hr": 150, "exercise_angina": 0, "oldpeak": 1.5, "st_slope": 1
                }
            ] * 1001  # Exceed limit
        }
        response = client.post('/api/predict/batch',
                              json=large_batch,
                              content_type='application/json')
        assert response.status_code in [400, 413]


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_minimum_age(self, client):
        """Test minimum age boundary."""
        patient = {
            "age": 20, "sex": 0, "chest_pain_type": 0, "resting_bp": 90,
            "cholesterol": 150, "fasting_bs": 0, "resting_ecg": 0,
            "max_hr": 180, "exercise_angina": 0, "oldpeak": 0, "st_slope": 0
        }
        response = client.post('/api/predict', json=patient)
        assert response.status_code == 200
    
    def test_maximum_age(self, client):
        """Test maximum age boundary."""
        patient = {
            "age": 80, "sex": 1, "chest_pain_type": 3, "resting_bp": 180,
            "cholesterol": 400, "fasting_bs": 1, "resting_ecg": 2,
            "max_hr": 100, "exercise_angina": 1, "oldpeak": 5, "st_slope": 2
        }
        response = client.post('/api/predict', json=patient)
        assert response.status_code == 200
    
    def test_zero_cholesterol(self, client):
        """Test zero cholesterol (edge case)."""
        patient = {
            "age": 40, "sex": 0, "chest_pain_type": 0, "resting_bp": 110,
            "cholesterol": 0, "fasting_bs": 0, "resting_ecg": 0,
            "max_hr": 160, "exercise_angina": 0, "oldpeak": 0, "st_slope": 0
        }
        response = client.post('/api/predict', json=patient)
        assert response.status_code == 200
    
    def test_high_cholesterol(self, client):
        """Test high cholesterol."""
        patient = {
            "age": 50, "sex": 1, "chest_pain_type": 1, "resting_bp": 150,
            "cholesterol": 400, "fasting_bs": 1, "resting_ecg": 2,
            "max_hr": 110, "exercise_angina": 1, "oldpeak": 4, "st_slope": 2
        }
        response = client.post('/api/predict', json=patient)
        assert response.status_code == 200
        data = json.loads(response.data)
        # High risk expected
        assert data['probability'] > 50
    
    def test_low_risk_profile(self, client):
        """Test low-risk patient profile."""
        patient = {
            "age": 30, "sex": 0, "chest_pain_type": 0, "resting_bp": 100,
            "cholesterol": 150, "fasting_bs": 0, "resting_ecg": 0,
            "max_hr": 190, "exercise_angina": 0, "oldpeak": 0, "st_slope": 0
        }
        response = client.post('/api/predict', json=patient)
        assert response.status_code == 200
        data = json.loads(response.data)
        # Low risk expected
        assert data['probability'] < 50


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Test performance and efficiency."""
    
    def test_single_prediction_speed(self, client, valid_input):
        """Test single prediction completes quickly."""
        import time
        start = time.time()
        response = client.post('/api/predict', json=valid_input)
        elapsed = time.time() - start
        assert elapsed < 0.5  # Should complete in less than 500ms
        assert response.status_code == 200
    
    def test_batch_prediction_speed(self, client):
        """Test batch prediction performs efficiently."""
        import time
        batch = {
            "patients": [
                {
                    "age": 45, "sex": 1, "chest_pain_type": 1, "resting_bp": 130,
                    "cholesterol": 240, "fasting_bs": 0, "resting_ecg": 0,
                    "max_hr": 150, "exercise_angina": 0, "oldpeak": 1.5, "st_slope": 1
                }
            ] * 100
        }
        start = time.time()
        response = client.post('/api/predict/batch', json=batch)
        elapsed = time.time() - start
        assert elapsed < 2.0  # 100 predictions should complete in 2 seconds
        assert response.status_code == 200


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
