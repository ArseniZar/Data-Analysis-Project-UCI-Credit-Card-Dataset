from sklearn.model_selection import KFold, StratifiedKFold
from custom_models import logistic_regression_gradient_descent, evaluate_logistic_regression, linear_regression_closed_form, evaluate_linear_regression
import numpy as np
import pandas as pd

# Walidacja krzyżowa dla regresji logistycznej
def cross_validate_logistic(X, y, n_splits=3):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w = logistic_regression_gradient_descent(X_train, y_train, learning_rate=0.01, epochs=100)
        metrics = evaluate_logistic_regression(X_train, y_train, X_val, y_val, X_val, y_val, w)
        metrics['Fold'] = fold + 1
        results.append(metrics)
    return pd.DataFrame(results)

# Walidacja krzyżowa dla regresji liniowej
def cross_validate_linear(X, y, n_splits=3):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w = linear_regression_closed_form(X_train, y_train)
        metrics = evaluate_linear_regression(X_train, y_train, X_val, y_val, X_val, y_val, w)
        metrics['Fold'] = fold + 1
        results.append(metrics)
    return pd.DataFrame(results)

# Przykład użycia (załóżmy, że X_class, y_class, X_reg, y_reg są zdefiniowane)
if __name__ == "__main__":
    # Dane przykładowe (zastąp swoimi danymi z main.py)
    from pipeline import train_and_evaluate_models, train_and_evaluate_regression
    from data_loader import DataLoader
    
    loader = DataLoader("data/UCI_Credit_Card.csv")
    data = loader.data
    X_class = data.drop(columns=["ID", "default.payment.next.month"]).values
    y_class = data["default.payment.next.month"].values
    X_reg = data.drop(columns=["ID", "LIMIT_BAL", "default.payment.next.month"]).values
    y_reg = data["LIMIT_BAL"].values
    
    # Klasyfikacja
    results_logistic = cross_validate_logistic(X_class, y_class)
    print("Cross-validation - Logistic Regression:")
    print(results_logistic)
    
    # Regresja
    results_linear = cross_validate_linear(X_reg, y_reg)
    print("Cross-validation - Linear Regression:")
    print(results_linear)