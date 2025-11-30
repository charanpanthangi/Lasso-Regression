import pandas as pd

from app.data import load_diabetes_dataset


def test_load_diabetes_dataset_shapes():
    X, y = load_diabetes_dataset()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 10  # diabetes dataset has 10 features
