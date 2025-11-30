import numpy as np

from app.evaluate import evaluate_regression


def test_evaluate_regression_metrics_are_non_negative():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])

    metrics = evaluate_regression(y_true, y_pred)

    assert metrics["mse"] >= 0
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert -1 <= metrics["r2"] <= 1
