"""Run the full Lasso regression workflow on the diabetes dataset."""

from pathlib import Path

from app.data import load_diabetes_dataset
from app.evaluate import evaluate_regression, format_metrics
from app.model import build_lasso_model, count_nonzero_coefficients, fit_model, predict
from app.preprocess import split_and_scale
from app.visualize import plot_actual_vs_predicted, plot_coefficients


def main() -> None:
    # 1. Load dataset
    X, y = load_diabetes_dataset()
    print(f"Loaded diabetes dataset with {X.shape[0]} rows and {X.shape[1]} features.")

    # 2. Split and scale data
    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
    ) = split_and_scale(X, y)

    # 3. Build and train model
    model = build_lasso_model(alpha=0.1, max_iter=5000)
    model = fit_model(model, X_train, y_train)

    # 4. Predict
    predictions = predict(model, X_test)

    # 5. Evaluate
    metrics = evaluate_regression(y_test, predictions)
    print("\nEvaluation metrics:\n" + format_metrics(metrics))

    # 6. Visualize
    actual_vs_predicted_path = plot_actual_vs_predicted(y_test, predictions)
    coefficients_path = plot_coefficients(X.columns, model.coef_)

    # 7. Report sparsity
    nonzero = count_nonzero_coefficients(model)
    print(f"\nNon-zero coefficients: {nonzero} / {len(model.coef_)}")
    print(f"Visualizations saved to: {Path(actual_vs_predicted_path).resolve()} and {Path(coefficients_path).resolve()}")


if __name__ == "__main__":
    main()
