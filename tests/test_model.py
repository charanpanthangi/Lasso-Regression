import numpy as np

from app.data import load_diabetes_dataset
from app.model import build_lasso_model, fit_model, predict
from app.preprocess import split_and_scale


def test_lasso_fit_and_predict_runs():
    X, y = load_diabetes_dataset()
    X_train, X_test, y_train, y_test, _ = split_and_scale(X, y, test_size=0.25)

    model = build_lasso_model(alpha=0.1)
    model = fit_model(model, X_train, y_train)
    preds = predict(model, X_test)

    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == y_test.shape[0]
