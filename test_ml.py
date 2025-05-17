import pytest
import numpy as np
import os
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.model import train_model, compute_model_metrics
from sklearn.ensemble import RandomForestClassifier

def test_train_model_returns_random_forest():
    """
    Test that train_model returns a RandomForestClassifier instance.
    """
    X = np.random.rand(10, 4)
    y = np.random.randint(0, 2, 10)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_compute_model_metrics_types():
    """
    Test that compute_model_metrics returns floats for precision, recall, and fbeta.
    """
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)

def test_actual_census_split():
    """
    Test the actual train/test split from census.csv as performed in train_model.py,
    accounting for sklearn's rounding (ceil).
    """

    data_path = os.path.join(os.getcwd(), "data", "census.csv")
    data = pd.read_csv(data_path)
    total_rows = len(data)

    train, test = train_test_split(data, test_size=0.20, random_state=42)

    # Use math.ceil to match sklearn's behavior
    expected_test_size = int(math.ceil(total_rows * 0.20)) #had to round to avoid off-by-one error :)
    expected_train_size = total_rows - expected_test_size

    assert len(test) == expected_test_size, f"Test set should have {expected_test_size} rows"
    assert len(train) == expected_train_size, f"Train set should have {expected_train_size} rows"
    assert len(train) + len(test) == total_rows, "Total rows should match original data"