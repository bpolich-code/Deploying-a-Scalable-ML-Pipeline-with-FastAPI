import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ml.model import (
    train_model,
    inference,
    compute_model_metrics,
    save_model,
    load_model
)
from ml.data import process_data
import os


@pytest.fixture
def sample_data():
    data = pd.read_csv('data/census.csv')
    return data


@pytest.fixture
def processed_data(sample_data):
    train, test = train_test_split(sample_data, test_size=0.20, random_state=42)
    
    cat_features = [
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native-country',
    ]
    
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label='salary',
        training=True
    )
    
    return X_train, y_train, encoder, lb


def test_train_model(processed_data):
    X_train, y_train, _, _ = processed_data
    model = train_model(X_train, y_train)
    
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, 'classes_')


def test_inference(processed_data):
    X_train, y_train, _, _ = processed_data
    model = train_model(X_train, y_train)
    
    preds = inference(model, X_train)
    
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_train.shape[0]
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics():
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 1])
    
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    
    assert isinstance(precision, (float, np.floating))
    assert isinstance(recall, (float, np.floating))
    assert isinstance(fbeta, (float, np.floating))
    
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1
    
    assert precision == 0.75
    assert recall == 0.75
    assert fbeta == 0.75


def test_save_and_load_model(processed_data, tmp_path):
    X_train, y_train, _, _ = processed_data
    model = train_model(X_train, y_train)
    
    model_path = tmp_path / 'test_model.pkl'
    save_model(model, str(model_path))
    
    assert model_path.exists()
    
    loaded_model = load_model(str(model_path))
    
    assert type(loaded_model) == type(model)
    
    preds_original = model.predict(X_train[:10])
    preds_loaded = loaded_model.predict(X_train[:10])
    assert np.array_equal(preds_original, preds_loaded)


def test_data_shape(sample_data):
    assert sample_data.shape[0] > 0
    assert 'salary' in sample_data.columns
    
    cat_features = [
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native-country',
    ]
    
    for feature in cat_features:
        assert feature in sample_data.columns
