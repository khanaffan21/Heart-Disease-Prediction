"""
Unit tests for model training and preprocessing.
"""

import pytest
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def sample_data():
    """Create sample heart disease data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'Age': np.random.randint(20, 80, n_samples),
        'Sex': np.random.choice(['M', 'F'], n_samples),
        'ChestPainType': np.random.choice(['ASY', 'ATA', 'NAP', 'TA'], n_samples),
        'RestingBP': np.random.randint(80, 200, n_samples),
        'Cholesterol': np.random.randint(0, 600, n_samples),
        'FastingBS': np.random.choice([0, 1], n_samples),
        'RestingECG': np.random.choice(['Normal', 'ST', 'LVH'], n_samples),
        'MaxHR': np.random.randint(60, 210, n_samples),
        'ExerciseAngina': np.random.choice(['Y', 'N'], n_samples),
        'Oldpeak': np.random.uniform(0, 6.2, n_samples),
        'ST_Slope': np.random.choice(['Up', 'Flat', 'Down'], n_samples),
        'HeartDisease': np.random.choice([0, 1], n_samples)
    }
    
    return pd.DataFrame(data)


def test_data_loading(sample_data):
    """Test data loading and inspection."""
    assert sample_data.shape[0] == 100
    assert sample_data.shape[1] == 12
    assert 'HeartDisease' in sample_data.columns
    assert sample_data.isnull().sum().sum() == 0


def test_data_shape(sample_data):
    """Test data shape after loading."""
    assert len(sample_data) > 0
    assert sample_data['HeartDisease'].isin([0, 1]).all()


def test_categorical_encoding(sample_data):
    """Test one-hot encoding of categorical features."""
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    X = sample_data.drop('HeartDisease', axis=1)
    
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop='first')
    
    # Check shape increased
    assert X_encoded.shape[1] > X.shape[1]
    # Check no missing values
    assert X_encoded.isnull().sum().sum() == 0


def test_feature_scaling(sample_data):
    """Test standardization of continuous features."""
    continuous_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
    X = sample_data[continuous_cols].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Check mean ≈ 0 and std ≈ 1
    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-10)


def test_train_test_split(sample_data):
    """Test train-test split."""
    from sklearn.model_selection import train_test_split
    
    X = sample_data.drop('HeartDisease', axis=1)
    y = sample_data['HeartDisease']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    assert len(X_train) == int(len(X) * 0.8)
    assert len(X_test) == int(len(X) * 0.2)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_model_training(sample_data):
    """Test logistic regression training."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Prepare data
    X = sample_data.drop('HeartDisease', axis=1).copy()
    y = sample_data['HeartDisease']
    
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    continuous_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
    
    X = pd.get_dummies(X, columns=categorical_cols, drop='first')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_train_scaled[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    
    X_test_scaled = X_test.copy()
    X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])
    
    # Train model
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver='lbfgs')
    model.fit(X_train_scaled, y_train)
    
    # Check model
    assert model.n_features_in_ == X_train_scaled.shape[1]
    assert model.coef_.shape == (1, X_train_scaled.shape[1])
    assert model.intercept_.shape == (1,)


def test_model_prediction(sample_data):
    """Test model prediction."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    X = sample_data.drop('HeartDisease', axis=1).copy()
    y = sample_data['HeartDisease']
    
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    continuous_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
    
    X = pd.get_dummies(X, columns=categorical_cols, drop='first')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_train_scaled[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    
    X_test_scaled = X_test.copy()
    X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])
    
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver='lbfgs')
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Check predictions
    assert y_pred.shape == y_test.shape
    assert y_pred_proba.shape == (len(y_test), 2)
    assert np.all((y_pred == 0) | (y_pred == 1))
    assert np.all((y_pred_proba >= 0) & (y_pred_proba <= 1))


def test_model_score(sample_data):
    """Test model score on test set."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    
    X = sample_data.drop('HeartDisease', axis=1).copy()
    y = sample_data['HeartDisease']
    
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    continuous_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
    
    X = pd.get_dummies(X, columns=categorical_cols, drop='first')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_train_scaled[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    
    X_test_scaled = X_test.copy()
    X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])
    
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver='lbfgs')
    model.fit(X_train_scaled, y_train)
    
    # Score
    score = model.score(X_test_scaled, y_test)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    assert score > 0
    assert score <= 1
    assert np.isclose(score, acc)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
