"""Preprocessing helpers for splitting and scaling data."""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Split the data into train and test sets and scale features.

    L1 regularization used by Lasso is sensitive to feature scaling because the
    penalty is applied to the magnitude of coefficients. Without scaling, one
    feature measured in a large unit could dominate the penalty term and
    distort which features are shrunk to zero. Standardizing to unit variance
    keeps the penalty fair across features.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train.to_numpy(), y_test.to_numpy(), scaler
