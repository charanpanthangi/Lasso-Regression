"""Visualization utilities for Lasso regression results."""

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(style="whitegrid")


EXAMPLES_DIR = Path("examples")


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    filename: Optional[str] = None,
) -> Path:
    """Plot actual vs. predicted values and save to the examples directory."""

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, color="#1f77b4")
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "--", color="black")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    fig.tight_layout()

    EXAMPLES_DIR.mkdir(exist_ok=True)
    output_path = EXAMPLES_DIR / (filename or "actual_vs_predicted.svg")
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def plot_coefficients(
    feature_names: Iterable[str],
    coefficients: np.ndarray,
    filename: Optional[str] = None,
) -> Path:
    """Plot coefficient magnitudes to illustrate sparsity."""

    coef_series = sorted(zip(feature_names, coefficients), key=lambda pair: abs(pair[1]), reverse=True)
    names, coefs = zip(*coef_series)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=list(names), y=list(coefs), ax=ax, palette="Blues_d")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Coefficient value")
    ax.set_title("Lasso Coefficients (some may be zero)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    EXAMPLES_DIR.mkdir(exist_ok=True)
    output_path = EXAMPLES_DIR / (filename or "lasso_coefficients.svg")
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path
