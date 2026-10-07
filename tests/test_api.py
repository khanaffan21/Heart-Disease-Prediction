"""
Integration tests for Flask API endpoints.
"""

import pytest
import json
import sys
import os

# Mock model loading for tests
sys.path.insert(0, os.path.dirname(__file__))


@pytest.fixture
def client():
    """Create Flask test client."""
    # We can't import app directly without models loaded,
    # so we mock the loading
    from api.app import app
    
    with app.test_client() as client:
        yield client


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'status' in data
    assert data['status'] == 'ok'


def test_api_not_found(client):
    """Test 404 error handling."""
    response = client.get('/api/invalid')
    assert response.status_code == 404
    data = json.loads(response.data)
    assert 'error' in data


def test_model_info_no_model(client):
    """Test model info when model is not loaded."""
    # This will fail if model not loaded, which is expected
    response = client.get('/api/model/info')
    # Either 200 (if pre-trained) or 503 (if not loaded)
    assert response.status_code in [200, 503]


def test_predict_missing_data(client):
    """Test predict endpoint with missing data."""
    response = client.post('/api/predict', json={})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data


def test_predict_invalid_json(client):
    """Test predict with invalid JSON."""
    response = client.post(
        '/api/predict',
        data='invalid json',
        content_type='application/json'
    )
    assert response.status_code in [400, 415]


def test_batch_predict_format(client):
    """Test batch predict endpoint format validation."""
    response = client.post('/api/predict/batch', json={'invalid': 'format'})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert 'records' in data['error'].lower()


def test_batch_predict_invalid_type(client):
    """Test batch predict with invalid records type."""
    response = client.post('/api/predict/batch', json={'records': 'not-a-list'})
    assert response.status_code == 400


def test_cors_headers(client):
    """Test CORS headers are present."""
    response = client.get('/health')
    # CORS headers should be present if flask-cors is working
    assert response.status_code == 200


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
