from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
from custom_models import (
    logistic_regression_gradient_descent,
    evaluate_logistic_regression,
    linear_regression_closed_form,
    evaluate_linear_regression,
)


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


def train_and_evaluate_models(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000)),
        ("Decision Tree", DecisionTreeClassifier(random_state=42)),
        ("SVM", SVC(kernel="rbf", random_state=42)),
    ]

    preprocessor = create_pipeline()
    results = []

    for name, model in models:
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)

        train_pred = pipeline.predict(X_train)
        val_pred = pipeline.predict(X_val)
        test_pred = pipeline.predict(X_test)

        results.append(
            {
                "Model": name,
                "Train Accuracy": accuracy_score(y_train, train_pred),
                "Train F1": f1_score(y_train, train_pred),
                "Validation Accuracy": accuracy_score(y_val, val_pred),
                "Validation F1": f1_score(y_val, val_pred),
                "Test Accuracy": accuracy_score(y_test, test_pred),
                "Test F1": f1_score(y_test, test_pred),
            }
        )

    X_train_transformed = preprocessor.fit_transform(X_train).toarray()
    X_val_transformed = preprocessor.transform(X_val).toarray()
    X_test_transformed = preprocessor.transform(X_test).toarray()

    w_custom = logistic_regression_gradient_descent(X_train_transformed, y_train.values)
    custom_results = evaluate_logistic_regression(
        X_train_transformed,
        y_train.values,
        X_val_transformed,
        y_val.values,
        X_test_transformed,
        y_test.values,
        w_custom,
    )

    results.append(
        {
            "Model": "Custom Logistic Regression",
            "Train Accuracy": custom_results["Accuracy Train"],
            "Train F1": custom_results["F1 Train"],
            "Validation Accuracy": custom_results["Accuracy Val"],
            "Validation F1": custom_results["F1 Val"],
            "Test Accuracy": custom_results["Accuracy Test"],
            "Test F1": custom_results["F1 Test"],
        }
    )

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

    X_reg_train, X_reg_temp, y_train, y_reg_temp = train_test_split(
        x_reg, y_reg, test_size=0.3, random_state=42
    )
    X_reg_val, X_reg_test, y_val, y_test = train_test_split(
        X_reg_temp, y_reg_temp, test_size=0.5, random_state=42
    )

    preprocessor = create_pipeline()
    X_train_transformed = preprocessor.fit_transform(X_reg_train).toarray()
    X_val_transformed = preprocessor.transform(X_reg_val).toarray()
    X_test_transformed = preprocessor.transform(X_reg_test).toarray()

    results = []
    w_reg = linear_regression_closed_form(X_train_transformed, y_train.values)
    custom_reg_results = evaluate_linear_regression(
        X_train_transformed,
        y_train.values,
        X_val_transformed,
        y_val.values,
        X_test_transformed,
        y_test.values,
        w_reg,
    )
    results.append(
        {
            "Model": "Custom Linear Regression",
            "MSE Train": custom_reg_results["MSE Train"],
            "MSE Val": custom_reg_results["MSE Val"],
            "MSE Test": custom_reg_results["MSE Test"],
        }
    )
    model_sk_reg = LinearRegression()
    model_sk_reg.fit(X_train_transformed, y_train)
    y_train_pred_sk = model_sk_reg.predict(X_train_transformed)
    y_val_pred_sk = model_sk_reg.predict(X_val_transformed)
    y_test_pred_sk = model_sk_reg.predict(X_test_transformed)
    mse_train_sk = np.mean((y_train - y_train_pred_sk) ** 2)
    mse_val_sk = np.mean((y_val - y_val_pred_sk) ** 2)
    mse_test_sk = np.mean((y_test - y_test_pred_sk) ** 2)
    results.append(
        {
            "Model": "SKLearn Linear Regression",
            "MSE Train": mse_train_sk,
            "MSE Val": mse_val_sk,
            "MSE Test": mse_test_sk,
        }
    )
    return pd.DataFrame(results)
