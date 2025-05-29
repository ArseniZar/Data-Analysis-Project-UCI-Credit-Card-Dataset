from pathlib import Path
from data_loader import DataLoader
from save_stats import save_stats_to_csv
from pipeline import train_and_evaluate_models, create_pipeline
from custom_models import (linear_regression_closed_form,evaluate_linear_regression,train_pytorch_model)
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import torch


def main():
    loader = DataLoader("data/UCI_Credit_Card.csv")
    if loader.data is None:
        print("Failed to load data. Check the file path.")
        return

    data = loader.data
    print("Missing values in data:\n", data.isna().sum())

   
    X_class = data.drop(columns=["ID", "default.payment.next.month"])
    y_class = data["default.payment.next.month"]
    (
        results_df_class,
        X_train_trans,
        X_val_trans,
        X_test_trans,
        y_train,
        y_val,
        y_test,
    ) = train_and_evaluate_models(X_class, y_class)

    # Train PyTorch model on CPU and GPU (if available)
    if torch.cuda.is_available():
        devices = ["cpu", "cuda"]
    else:
        devices = ["cpu"]

    for device in devices:
        print(f"Training PyTorch model on {device}")
        results_pytorch = train_pytorch_model(
            X_train_trans,
            y_train.values,
            X_val_trans,
            y_val.values,
            X_test_trans,
            y_test.values,
            device=device,
        )
        pytorch_row = {
            "Model": f"PyTorch Logistic Regression ({device})",
            "Train Accuracy": results_pytorch["Accuracy Train"],
            "Train F1": results_pytorch["F1 Train"],
            "Validation Accuracy": results_pytorch["Accuracy Val"],
            "Validation F1": results_pytorch["F1 Val"],
            "Test Accuracy": results_pytorch["Accuracy Test"],
            "Test F1": results_pytorch["F1 Test"],
        }
        results_df_class = pd.concat(
            [results_df_class, pd.DataFrame([pytorch_row])], ignore_index=True
        )
        if device == "cpu":
            time_cpu = results_pytorch["training_time"]
        elif device == "cuda":
            time_gpu = results_pytorch["training_time"]

    if torch.cuda.is_available():
        print(f"Training time on CPU: {time_cpu:.2f} seconds")
        print(f"Training time on GPU: {time_gpu:.2f} seconds")
        print(f"Speedup: {time_cpu / time_gpu:.2f}x")

    # Regression
    X_reg = data.drop(columns=["ID", "LIMIT_BAL", "default.payment.next.month"])
    y_reg = data["LIMIT_BAL"]
    from sklearn.model_selection import train_test_split

    X_reg_train, X_reg_temp, y_reg_train, y_reg_temp = train_test_split(
        X_reg, y_reg, test_size=0.3, random_state=42
    )
    X_reg_val, X_reg_test, y_reg_val, y_reg_test = train_test_split(
        X_reg_temp, y_reg_temp, test_size=0.5, random_state=42
    )

    preprocessor = create_pipeline()
    X_reg_train_transformed = preprocessor.fit_transform(X_reg_train).toarray()
    X_reg_val_transformed = preprocessor.transform(X_reg_val).toarray()
    X_reg_test_transformed = preprocessor.transform(X_reg_test).toarray()

    w_reg = linear_regression_closed_form(X_reg_train_transformed, y_reg_train.values)
    custom_reg_results = evaluate_linear_regression(
        X_reg_train_transformed,
        y_reg_train.values,
        X_reg_val_transformed,
        y_reg_val.values,
        X_reg_test_transformed,
        y_reg_test.values,
        w_reg,
    )

    model_sk_reg = LinearRegression()
    model_sk_reg.fit(X_reg_train_transformed, y_reg_train)
    y_train_pred_sk = model_sk_reg.predict(X_reg_train_transformed)
    y_val_pred_sk = model_sk_reg.predict(X_reg_val_transformed)
    y_test_pred_sk = model_sk_reg.predict(X_reg_test_transformed)

    mse_train_sk = np.mean((y_reg_train - y_train_pred_sk) ** 2)
    mse_val_sk = np.mean((y_reg_val - y_val_pred_sk) ** 2)
    mse_test_sk = np.mean((y_reg_test - y_test_pred_sk) ** 2)

    results_reg = pd.DataFrame(
        [
            {
                "Model": "Custom Linear Regression",
                "MSE Train": custom_reg_results["MSE Train"],
                "MSE Val": custom_reg_results["MSE Val"],
                "MSE Test": custom_reg_results["MSE Test"],
            },
            {
                "Model": "SKLearn Linear Regression",
                "MSE Train": mse_train_sk,
                "MSE Val": mse_val_sk,
                "MSE Test": mse_test_sk,
            },
        ]
    )

    Path("Part-2/results").mkdir(exist_ok=True)
    filename_class = Path("Part-2/results") / "classification_results.csv"
    filename_reg = Path("Part-2/results") / "regression_results.csv"

    if save_stats_to_csv(results_df_class, filename_class):
        print(f"Classification results saved to {filename_class}")
    if save_stats_to_csv(results_reg, filename_reg):
        print(f"Regression results saved to {filename_reg}")

    print("\nClassification results table:")
    print(results_df_class)
    print("\nRegression results table:")
    print(results_reg)


if __name__ == "__main__":
    main()
