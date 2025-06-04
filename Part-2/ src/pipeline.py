from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import pandas as pd
import numpy as np
from custom_models import (
    logistic_regression_gradient_descent,
    evaluate_logistic_regression,
    linear_regression_closed_form,
    linear_regression_gradient_descent,
    evaluate_linear_regression,
    predict_logistic_regression,
    predict_linear_regression,
)
import matplotlib.pyplot as plt
from pathlib import Path

def create_pipeline():
    numeric_features = [
        "AGE",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]

    categorical_features = [
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor

def perform_cv(model, X, y, cv=3, is_classification=True):
    if is_classification:
        skf = StratifiedKFold(n_splits=cv)
    else:
        skf = KFold(n_splits=cv)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        preprocessor = create_pipeline()
        X_train_transformed = preprocessor.fit_transform(X_train).toarray()
        X_val_transformed = preprocessor.transform(X_val).toarray()
        if isinstance(model, str):  # Для кастомных моделей
            if model == 'custom_logistic':
                w = logistic_regression_gradient_descent(X_train_transformed, y_train.values)
                y_pred = predict_logistic_regression(X_val_transformed, w)
            elif model == 'custom_linear':
                w, _, _ = linear_regression_gradient_descent(X_train_transformed, y_train.values, 
                                                            X_val_transformed, y_val.values)
                y_pred = predict_linear_regression(X_val_transformed, w)
        else:  # Для sklearn моделей
            model.fit(X_train_transformed, y_train)
            y_pred = model.predict(X_val_transformed)
        if is_classification:
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            scores.append({'accuracy': acc, 'f1': f1})
        else:
            mse = mean_squared_error(y_val, y_pred)
            scores.append({'mse': mse})
    if is_classification:
        mean_acc = np.mean([s['accuracy'] for s in scores])
        std_acc = np.std([s['accuracy'] for s in scores])
        mean_f1 = np.mean([s['f1'] for s in scores])
        std_f1 = np.std([s['f1'] for s in scores])
        return {'CV Accuracy Mean': mean_acc, 'CV Accuracy Std': std_acc, 
                'CV F1 Mean': mean_f1, 'CV F1 Std': std_f1}
    else:
        mean_mse = np.mean([s['mse'] for s in scores])
        std_mse = np.std([s['mse'] for s in scores])
        return {'CV MSE Mean': mean_mse, 'CV MSE Std': std_mse}

def train_and_evaluate_models(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000)),
        ("Decision Tree", DecisionTreeClassifier(random_state=42)),
        ("SVM", SVC(kernel="rbf", random_state=42)),
        ("Custom Logistic Regression", "custom_logistic"),
        ("Custom Logistic Regression (L2)", "custom_logistic"),  # С регуляризацией
    ]

    preprocessor = create_pipeline()
    X_train_transformed = preprocessor.fit_transform(X_train).toarray()
    X_val_transformed = preprocessor.transform(X_val).toarray()
    X_test_transformed = preprocessor.transform(X_test).toarray()

    results = []

    for model_name, model in models:
        if model == "custom_logistic":
            # Без регуляризации
            cv_results = perform_cv(model, X_train, y_train)
            w = logistic_regression_gradient_descent(X_train_transformed, y_train.values, reg_lambda=0.0)
            custom_results = evaluate_logistic_regression(
                X_train_transformed, y_train.values, X_val_transformed, y_val.values, 
                X_test_transformed, y_test.values, w
            )
            results.append({
                "Model": "Custom Logistic Regression",
                "CV Accuracy Mean": cv_results['CV Accuracy Mean'],
                "CV Accuracy Std": cv_results['CV Accuracy Std'],
                "CV F1 Mean": cv_results['CV F1 Mean'],
                "CV F1 Std": cv_results['CV F1 Std'],
                "Train Accuracy": custom_results['Accuracy Train'],
                "Train F1": custom_results['F1 Train'],
                "Validation Accuracy": custom_results['Accuracy Val'],
                "Validation F1": custom_results['F1 Val'],
                "Test Accuracy": custom_results['Accuracy Test'],
                "Test F1": custom_results['F1 Test'],
            })
        elif model_name == "Custom Logistic Regression (L2)":
            # С L2 регуляризацией
            cv_results = perform_cv(model, X_train, y_train)
            w = logistic_regression_gradient_descent(X_train_transformed, y_train.values, reg_lambda=0.01)
            custom_results = evaluate_logistic_regression(
                X_train_transformed, y_train.values, X_val_transformed, y_val.values, 
                X_test_transformed, y_test.values, w
            )
            results.append({
                "Model": "Custom Logistic Regression (L2)",
                "CV Accuracy Mean": cv_results['CV Accuracy Mean'],
                "CV Accuracy Std": cv_results['CV Accuracy Std'],
                "CV F1 Mean": cv_results['CV F1 Mean'],
                "CV F1 Std": cv_results['CV F1 Std'],
                "Train Accuracy": custom_results['Accuracy Train'],
                "Train F1": custom_results['F1 Train'],
                "Validation Accuracy": custom_results['Accuracy Val'],
                "Validation F1": custom_results['F1 Val'],
                "Test Accuracy": custom_results['Accuracy Test'],
                "Test F1": custom_results['F1 Test'],
            })
        else:
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
            cv_results = perform_cv(model, X_train, y_train)
            pipeline.fit(X_train, y_train)
            train_pred = pipeline.predict(X_train)
            val_pred = pipeline.predict(X_val)
            test_pred = pipeline.predict(X_test)
            results.append({
                "Model": model_name,
                "CV Accuracy Mean": cv_results['CV Accuracy Mean'],
                "CV Accuracy Std": cv_results['CV Accuracy Std'],
                "CV F1 Mean": cv_results['CV F1 Mean'],
                "CV F1 Std": cv_results['CV F1 Std'],
                "Train Accuracy": accuracy_score(y_train, train_pred),
                "Train F1": f1_score(y_train, train_pred),
                "Validation Accuracy": accuracy_score(y_val, val_pred),
                "Validation F1": f1_score(y_val, val_pred),
                "Test Accuracy": accuracy_score(y_test, test_pred),
                "Test F1": f1_score(y_test, test_pred),
            })

    return (
        pd.DataFrame(results),
        X_train_transformed,
        X_val_transformed,
        X_test_transformed,
        y_train,
        y_val,
        y_test,
    )

def train_and_evaluate_regression(x_reg, y_reg):
    X_reg_train, X_reg_temp, y_reg_train, y_reg_temp = train_test_split(
        x_reg, y_reg, test_size=0.3, random_state=42
    )
    X_reg_val, X_reg_test, y_reg_val, y_reg_test = train_test_split(
        X_reg_temp, y_reg_temp, test_size=0.5, random_state=42
    )

    preprocessor = create_pipeline()
    X_train_transformed = preprocessor.fit_transform(X_reg_train).toarray()
    X_val_transformed = preprocessor.transform(X_reg_val).toarray()
    X_test_transformed = preprocessor.transform(X_reg_test).toarray()

    results = []

    # Custom Linear Regression (без регуляризации)
    w, train_costs, val_costs = linear_regression_gradient_descent(
        X_train_transformed, y_reg_train.values, X_val_transformed, y_reg_val.values, reg_lambda=0.0
    )
    custom_reg_results = evaluate_linear_regression(
        X_train_transformed, y_reg_train.values, X_val_transformed, y_reg_val.values, 
        X_test_transformed, y_reg_test.values, w
    )
    cv_results = perform_cv("custom_linear", X_reg_train, y_reg_train, is_classification=False)
    results.append({
        "Model": "Custom Linear Regression",
        "CV MSE Mean": cv_results['CV MSE Mean'],
        "CV MSE Std": cv_results['CV MSE Std'],
        "MSE Train": custom_reg_results["MSE Train"],
        "MSE Val": custom_reg_results["MSE Val"],
        "MSE Test": custom_reg_results["MSE Test"],
    })

    # Custom Linear Regression (L2)
    w_l2, train_costs_l2, val_costs_l2 = linear_regression_gradient_descent(
        X_train_transformed, y_reg_train.values, X_val_transformed, y_reg_val.values, reg_lambda=0.01
    )
    custom_reg_results_l2 = evaluate_linear_regression(
        X_train_transformed, y_reg_train.values, X_val_transformed, y_reg_val.values, 
        X_test_transformed, y_reg_test.values, w_l2
    )
    cv_results_l2 = perform_cv("custom_linear", X_reg_train, y_reg_train, is_classification=False)
    results.append({
        "Model": "Custom Linear Regression (L2)",
        "CV MSE Mean": cv_results_l2['CV MSE Mean'],
        "CV MSE Std": cv_results_l2['CV MSE Std'],
        "MSE Train": custom_reg_results_l2["MSE Train"],
        "MSE Val": custom_reg_results_l2["MSE Val"],
        "MSE Test": custom_reg_results_l2["MSE Test"],
    })

    # Построение графика сходимости
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_costs)), train_costs, label='Train MSE (no reg)')
    plt.plot(range(len(val_costs)), val_costs, label='Val MSE (no reg)')
    plt.plot(range(len(train_costs_l2)), train_costs_l2, label='Train MSE (L2)', linestyle='--')
    plt.plot(range(len(val_costs_l2)), val_costs_l2, label='Val MSE (L2)', linestyle='--')
    plt.xlabel('Эпоха')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('График сходимости для Custom Linear Regression')
    plt.savefig('Part-2/results/convergence_plot.png')
    plt.close()

    # Sklearn Linear Regression
    model_sk_reg = LinearRegression()
    model_sk_reg.fit(X_train_transformed, y_reg_train)
    y_train_pred_sk = model_sk_reg.predict(X_train_transformed)
    y_val_pred_sk = model_sk_reg.predict(X_val_transformed)
    y_test_pred_sk = model_sk_reg.predict(X_test_transformed)
    mse_train_sk = np.mean((y_reg_train - y_train_pred_sk) ** 2)
    mse_val_sk = np.mean((y_reg_val - y_val_pred_sk) ** 2)
    mse_test_sk = np.mean((y_reg_test - y_test_pred_sk) ** 2)
    cv_results_sk = perform_cv(model_sk_reg, X_reg_train, y_reg_train, is_classification=False)
    results.append({
        "Model": "SKLearn Linear Regression",
        "CV MSE Mean": cv_results_sk['CV MSE Mean'],
        "CV MSE Std": cv_results_sk['CV MSE Std'],
        "MSE Train": mse_train_sk,
        "MSE Val": mse_val_sk,
        "MSE Test": mse_test_sk,
    })

    # Ridge Regression
    model_ridge = Ridge(alpha=1.0)
    model_ridge.fit(X_train_transformed, y_reg_train)
    y_train_pred_ridge = model_ridge.predict(X_train_transformed)
    y_val_pred_ridge = model_ridge.predict(X_val_transformed)
    y_test_pred_ridge = model_ridge.predict(X_test_transformed)
    mse_train_ridge = np.mean((y_reg_train - y_train_pred_ridge) ** 2)
    mse_val_ridge = np.mean((y_reg_val - y_val_pred_ridge) ** 2)
    mse_test_ridge = np.mean((y_reg_test - y_test_pred_ridge) ** 2)
    cv_results_ridge = perform_cv(model_ridge, X_reg_train, y_reg_train, is_classification=False)
    results.append({
        "Model": "Ridge Regression",
        "CV MSE Mean": cv_results_ridge['CV MSE Mean'],
        "CV MSE Std": cv_results_ridge['CV MSE Std'],
        "MSE Train": mse_train_ridge,
        "MSE Val": mse_val_ridge,
        "MSE Test": mse_test_ridge,
    })

    # Lasso Regression
    model_lasso = Lasso(alpha=0.1)
    model_lasso.fit(X_train_transformed, y_reg_train)
    y_train_pred_lasso = model_lasso.predict(X_train_transformed)
    y_val_pred_lasso = model_lasso.predict(X_val_transformed)
    y_test_pred_lasso = model_lasso.predict(X_test_transformed)
    mse_train_lasso = np.mean((y_reg_train - y_train_pred_lasso) ** 2)
    mse_val_lasso = np.mean((y_reg_val - y_val_pred_lasso) ** 2)
    mse_test_lasso = np.mean((y_reg_test - y_test_pred_lasso) ** 2)
    cv_results_lasso = perform_cv(model_lasso, X_reg_train, y_reg_train, is_classification=False)
    results.append({
        "Model": "Lasso Regression",
        "CV MSE Mean": cv_results_lasso['CV MSE Mean'],
        "CV MSE Std": cv_results_lasso['CV MSE Std'],
        "MSE Train": mse_train_lasso,
        "MSE Val": mse_val_lasso,
        "MSE Test": mse_test_lasso,
    })

    return pd.DataFrame(results)