# Lasso Regression Template (scikit-learn)

A beginner-friendly, end-to-end template showing how to train, evaluate, and visualize a Lasso regression model on the scikit-learn diabetes dataset. Lasso (L1 regularization) can shrink some coefficients exactly to zero, which makes it useful for feature selection.

## What is Lasso Regression?

Lasso adds an L1 penalty to the loss function. The penalty discourages large coefficients and can set some coefficients to zero, effectively performing feature selection. Compared with Ridge (L2), Lasso tends to produce sparse models; higher `alpha` values increase sparsity. Scaling features (e.g., with `StandardScaler`) is critical so the penalty treats each feature fairly.

## When to Use It

- You want a linear model that can automatically drop unimportant features.
- You have many correlated or noisy predictors.
- You prefer a simple, interpretable model where zeroed coefficients indicate irrelevant features.

## Dataset

We use the built-in scikit-learn **diabetes** dataset (442 rows, 10 numeric predictors). The target is a continuous diabetes progression measure.

## Project Structure

```
app/
  data.py         # Load diabetes dataset
  preprocess.py   # Train/test split + scaling (crucial for L1 regularization)
  model.py        # Build and fit Lasso model
  evaluate.py     # Common regression metrics
  visualize.py    # Simple plots for predictions and coefficients
  main.py         # Full training + evaluation pipeline
notebooks/
  demo_lasso_regression.ipynb  # Interactive walkthrough
examples/        # Saved SVG plots
requirements.txt
Dockerfile
LICENSE
README.md
```

## How the Pipeline Works

1. **Load data** with `load_diabetes_dataset`.
2. **Split & scale** using `split_and_scale` (StandardScaler keeps the L1 penalty balanced across features).
3. **Build model** with `build_lasso_model(alpha=0.1)`.
4. **Fit & predict** using `fit_model` and `predict`.
5. **Evaluate** via `evaluate_regression` (MSE, MAE, RMSE, R²).
6. **Visualize** predicted vs. actual values and coefficient magnitudes (showing sparsity).

## Running the Script

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app/main.py
```

SVG plots are written to `examples/`.

## Jupyter Notebook

Launch the notebook for an interactive explanation:

```bash
jupyter notebook notebooks/demo_lasso_regression.ipynb
```

## Future Extensions

- Compare **Ridge vs Lasso** side by side.
- Use **LassoCV** for cross-validated alpha selection.
- Add hyperparameter search for better generalization.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
