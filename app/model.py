"""Model builder for Lasso regression."""

from typing import Tuple

import numpy as np
from sklearn.linear_model import Lasso


def build_lasso_model(alpha: float = 0.1, max_iter: int = 5000) -> Lasso:
    """Create a configured scikit-learn Lasso model.

    Parameters
    ----------
    alpha: float
        Regularization strength. Higher values increase sparsity by pushing
        more coefficients to zero.
    max_iter: int
        Maximum iterations for the coordinate descent solver.
    """

    return Lasso(alpha=alpha, max_iter=max_iter)


def fit_model(model: Lasso, X_train: np.ndarray, y_train: np.ndarray) -> Lasso:
    """Fit the Lasso model and return it for convenience."""

    model.fit(X_train, y_train)
    return model


def predict(model: Lasso, X_test: np.ndarray) -> np.ndarray:
    """Generate predictions using the fitted model."""

    return model.predict(X_test)


def count_nonzero_coefficients(model: Lasso) -> int:
    """Count how many coefficients are non-zero to illustrate sparsity."""

    return int(np.count_nonzero(model.coef_))
