"""Evaluation helpers for regression models."""

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute common regression metrics for convenience."""

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


def format_metrics(metrics: Dict[str, float]) -> str:
    """Create a neat multi-line string of the metrics."""

    lines = [
        f"Mean Squared Error: {metrics['mse']:.3f}",
        f"Mean Absolute Error: {metrics['mae']:.3f}",
        f"Root Mean Squared Error: {metrics['rmse']:.3f}",
        f"R^2 Score: {metrics['r2']:.3f}",
    ]
    return "\n".join(lines)
