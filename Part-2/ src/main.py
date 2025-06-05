from pathlib import Path
from data_loader import DataLoader
from save_stats import save_stats_to_csv
from pipeline import train_and_evaluate_models, train_and_evaluate_regression, train_and_evaluate_models_balanced, tune_decision_tree, tune_svc
from custom_models import train_pytorch_model
import pandas as pd
import torch

def main():
    print("Запуск программы...")
    loader = DataLoader("data/UCI_Credit_Card.csv")
    if loader.data is None:
        print("Не удалось загрузить данные. Проверьте путь к файлу.")
        return

    print("Данные загружены успешно.")
    data = loader.data
    print("Пропущенные значения в данных:\n", data.isna().sum())

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

    # Часть III: 3.4 Балансировка данных
    print("\n=== Часть III: 3.4 Балансировка данных ===")
    results_df_class_balanced = train_and_evaluate_models_balanced(X_class, y_class)
    print("Результаты классификации на сбалансированных данных:")
    print(results_df_class_balanced)

    # Часть III: 3.5 Оптимизация гиперпараметров
    print("\n=== Часть III: 3.5 Оптимизация гиперпараметров ===")
    print("Оптимизация Decision Tree...")
    best_params_dt, best_score_dt = tune_decision_tree(X_train_trans, y_train)
    print(f"Лучшие параметры для Decision Tree: {best_params_dt}, Лучший F1: {best_score_dt:.4f}")

    print("Оптимизация SVC...")
    best_params_svc, best_score_svc = tune_svc(X_train_trans, y_train)
    print(f"Лучшие параметры для SVC: {best_params_svc}, Лучший F1: {best_score_svc:.4f}")

    # Часть II: Обучение PyTorch модели
    print("\n=== Часть II: Обучение PyTorch модели ===")
    if torch.cuda.is_available():
        devices = ["cpu", "cuda"]
    else:
        devices = ["cpu"]

    for device in devices:
        print(f"Обучение PyTorch модели на {device}...")
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
        print(f"Время обучения на CPU: {time_cpu:.2f} секунд")
        print(f"Время обучения на GPU: {time_gpu:.2f} секунд")
        print(f"Ускорение: {time_cpu / time_gpu:.2f}x")

    X_reg = data.drop(columns=["ID", "LIMIT_BAL", "default.payment.next.month"])
    y_reg = data["default.payment.next.month"]
    results_reg = train_and_evaluate_regression(X_reg, y_reg)

    print("Сохранение результатов...")
    Path("Part-2/results").mkdir(exist_ok=True)
    filename_class = Path("Part-2/results") / "classification_results.csv"
    filename_reg = Path("Part-2/results") / "regression_results.csv"
    filename_class_balanced = Path("Part-2/results") / "classification_results_balanced.csv"

    if save_stats_to_csv(results_df_class, filename_class):
        print(f"Результаты классификации сохранены в {filename_class}")
    if save_stats_to_csv(results_reg, filename_reg):
        print(f"Результаты регрессии сохранены в {filename_reg}")
    if save_stats_to_csv(results_df_class_balanced, filename_class_balanced):
        print(f"Результаты классификации на сбалансированных данных сохранены в {filename_class_balanced}")

    print("\nТаблица результатов классификации:")
    print(results_df_class)
    print("\nТаблица результатов регрессии:")
    print(results_reg)

    print("Программа завершена успешно.")

if __name__ == "__main__":
    main()