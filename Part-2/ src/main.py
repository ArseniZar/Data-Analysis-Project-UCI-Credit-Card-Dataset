from pathlib import Path
from data_loader import DataLoader
from save_stats import save_stats_to_csv
from pipeline import train_and_evaluate_models, train_and_evaluate_regression
from custom_models import train_pytorch_model
import pandas as pd
import torch

def main():
    print("Starting main execution...")
    loader = DataLoader("data/UCI_Credit_Card.csv")
    if loader.data is None:
        print("Failed to load data. Check the file path.")
        return

    data = loader.data
    print("Missing values in data:\n", data.isna().sum())

    # Classification
    X_class = data.drop(columns=["ID", "default.payment.next.month"])
    y_class = data["default.payment.next.month"]
    print("Starting classification training and evaluation...")
    (
        results_df_class,
        X_train_trans,
        X_val_trans,
        X_test_trans,
        y_train,
        y_val,
        y_test,
    ) = train_and_evaluate_models(X_class, y_class)

    # PyTorch model
    devices = ["cpu"] if not torch.cuda.is_available() else ["cpu", "cuda"]
    for device in devices:
        print(f"Training PyTorch model on {device}...")
        results_pytorch = train_pytorch_model(
            X_train_trans, y_train.values, X_val_trans, y_val.values, X_test_trans, y_test.values, device=device
        )
        pytorch_row = {
            "Model": f"PyTorch Logistic Regression ({device})",
            "CV Accuracy Mean": None,
            "CV Accuracy Std": None,
            "CV F1 Mean": None,
            "CV F1 Std": None,
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
    print("Starting regression training and evaluation...")
    results_reg = train_and_evaluate_regression(X_reg, y_reg)

    Path("Part-2/results").mkdir(exist_ok=True)
    filename_class = Path("Part-2/results") / "classification_results.csv"
    filename_reg = Path("Part-2/results") / "regression_results.csv"

    save_stats_to_csv(results_df_class, filename_class)
    save_stats_to_csv(results_reg, filename_reg)

    print("\nClassification results table:")
    print(results_df_class)
    print("\nRegression results table:")
    print(results_reg)
    print("Main execution completed.")

if __name__ == "__main__":
    main()