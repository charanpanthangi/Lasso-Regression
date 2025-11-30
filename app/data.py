"""Data loading utilities for the Lasso regression example."""

from typing import Tuple

import pandas as pd
from sklearn.datasets import load_diabetes


def load_diabetes_dataset(as_frame: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the scikit-learn diabetes dataset.

    Parameters
    ----------
    as_frame: bool
        Whether to return the data as pandas objects. The default is True for
        beginner-friendly inspection and plotting.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Feature matrix ``X`` and target vector ``y``.
    """

    dataset = load_diabetes(as_frame=as_frame)
    X = dataset.data
    y = dataset.target
    return X, y
