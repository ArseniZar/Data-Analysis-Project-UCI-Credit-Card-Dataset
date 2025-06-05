from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, precision_score, recall_score
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
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_regression  # Added for feature selection

def create_pipeline():
    numeric_features = [
        "AGE", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    ]
    categorical_features = [
        "SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
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

def perform_cv(model, X, y, cv=3, is_classification=True, use_smote=False):
    print(f"Performing {cv}-fold cross-validation for {model} with use_smote={use_smote}...")
    if is_classification:
        skf = StratifiedKFold(n_splits=cv)
    else:
        skf = KFold(n_splits=cv)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"Processing fold {fold}/{cv}...")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        preprocessor = create_pipeline()
        X_train_transformed = preprocessor.fit_transform(X_train).toarray()
        X_val_transformed = preprocessor.transform(X_val).toarray()
        if use_smote and is_classification:
            smote = SMOTE(random_state=42)
            X_train_transformed, y_train_res = smote.fit_resample(X_train_transformed, y_train)
        else:
            y_train_res = y_train
        if isinstance(model, str):
            if model == 'custom_logistic':
                w = logistic_regression_gradient_descent(X_train_transformed, y_train_res.values)
                y_pred = predict_logistic_regression(X_val_transformed, w)
            elif model == 'custom_linear':
                w, _, _ = linear_regression_gradient_descent(X_train_transformed, y_train_res.values, 
                                                            X_val_transformed, y_val.values)
                y_pred = predict_linear_regression(X_val_transformed, w)
        else:
            model.fit(X_train_transformed, y_train_res)
            y_pred = model.predict(X_val_transformed)
        if is_classification:
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            scores.append({'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall})
        else:
            mse = mean_squared_error(y_val, y_pred)
            scores.append({'mse': mse})
    if is_classification:
        mean_acc = np.mean([s['accuracy'] for s in scores])
        std_acc = np.std([s['accuracy'] for s in scores])
        mean_f1 = np.mean([s['f1'] for s in scores])
        std_f1 = np.std([s['f1'] for s in scores])
        mean_precision = np.mean([s['precision'] for s in scores])
        std_precision = np.std([s['precision'] for s in scores])
        mean_recall = np.mean([s['recall'] for s in scores])
        std_recall = np.std([s['recall'] for s in scores])
        print(f"CV completed: Acc Mean={mean_acc:.4f}, F1 Mean={mean_f1:.4f}, Precision Mean={mean_precision:.4f}, Recall Mean={mean_recall:.4f}")
        return {
            'CV Accuracy Mean': mean_acc, 'CV Accuracy Std': std_acc,
            'CV F1 Mean': mean_f1, 'CV F1 Std': std_f1,
            'CV Precision Mean': mean_precision, 'CV Precision Std': std_precision,
            'CV Recall Mean': mean_recall, 'CV Recall Std': std_recall
        }
    else:
        mean_mse = np.mean([s['mse'] for s in scores])
        std_mse = np.std([s['mse'] for s in scores])
        print(f"CV completed: MSE Mean={mean_mse:.4f}")
        return {'CV MSE Mean': mean_mse, 'CV MSE Std': std_mse}

def train_and_evaluate_models(X, y):
    print("Splitting data for classification...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000)),
        ("Decision Tree", DecisionTreeClassifier(random_state=42)),
        ("SVM", SVC(kernel="rbf", random_state=42)),
        ("Custom Logistic Regression", "custom_logistic"),
        ("Custom Logistic Regression (L2)", "custom_logistic"),
    ]

    preprocessor = create_pipeline()
    print("Preprocessing data...")
    X_train_transformed = preprocessor.fit_transform(X_train).toarray()
    X_val_transformed = preprocessor.transform(X_val).toarray()
    X_test_transformed = preprocessor.transform(X_test).toarray()

    results = []

    for model_name, model in models:
        reg_lambda = 0.01 if model_name == "Custom Logistic Regression (L2)" else 0.0
        model_type = model if not isinstance(model, str) else "custom_logistic"
        
        for use_smote in [False, True]:
            name = f"{model_name} (SMOTE)" if use_smote else model_name
            print(f"Evaluating {name}...")
            cv_results = perform_cv(model_type, X_train, y_train, cv=3, is_classification=True, use_smote=use_smote)
            if use_smote:
                smote = SMOTE(random_state=42)
                X_train_current, y_train_current = smote.fit_resample(X_train_transformed, y_train)
            else:
                X_train_current, y_train_current = X_train_transformed, y_train
            if isinstance(model_type, str):
                w = logistic_regression_gradient_descent(X_train_current, y_train_current.values, reg_lambda=reg_lambda)
                y_train_pred = predict_logistic_regression(X_train_transformed, w)
                y_val_pred = predict_logistic_regression(X_val_transformed, w)
                y_test_pred = predict_logistic_regression(X_test_transformed, w)
            else:
                model.fit(X_train_current, y_train_current)
                y_train_pred = model.predict(X_train_transformed)
                y_val_pred = model.predict(X_val_transformed)
                y_test_pred = model.predict(X_test_transformed)
            train_acc = accuracy_score(y_train, y_train_pred)
            train_f1 = f1_score(y_train, y_train_pred)
            val_acc = accuracy_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred)
            results.append({
                "Model": name,
                "CV Accuracy Mean": cv_results['CV Accuracy Mean'],
                "CV Accuracy Std": cv_results['CV Accuracy Std'],
                "CV F1 Mean": cv_results['CV F1 Mean'],
                "CV F1 Std": cv_results['CV F1 Std'],
                "CV Precision Mean": cv_results['CV Precision Mean'],
                "CV Precision Std": cv_results['CV Precision Std'],
                "CV Recall Mean": cv_results['CV Recall Mean'],
                "CV Recall Std": cv_results['CV Recall Std'],
                "Train Accuracy": train_acc,
                "Train F1": train_f1,
                "Validation Accuracy": val_acc,
                "Validation F1": val_f1,
                "Test Accuracy": test_acc,
                "Test F1": test_f1,
            })

    tuned_models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000), {'C': [0.01, 0.1, 1, 10]}),
        ("Decision Tree", DecisionTreeClassifier(random_state=42), {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5]}),
    ]
    for model_name, model, param_grid in tuned_models:
        print(f"Performing grid search for {model_name}...")
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1', n_jobs=1)
        grid_search.fit(X_train_transformed, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        y_val_pred = best_model.predict(X_val_transformed)
        y_test_pred = best_model.predict(X_test_transformed)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        results.append({
            "Model": f"{model_name} (tuned)",
            "Best Params": best_params,
            "CV F1 Mean": best_score,
            "Validation Accuracy": val_acc,
            "Validation F1": val_f1,
            "Test Accuracy": test_acc,
            "Test F1": test_f1,
        })
        print(f"Grid search for {model_name} completed. Best params: {best_params}")

    print("Classification evaluation completed.")
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
    print("Splitting data for regression...")
    X_reg_train, X_reg_temp, y_reg_train, y_reg_temp = train_test_split(
        x_reg, y_reg, test_size=0.3, random_state=42
    )
    X_reg_val, X_reg_test, y_reg_val, y_reg_test = train_test_split(
        X_reg_temp, y_reg_temp, test_size=0.5, random_state=42
    )

    preprocessor = create_pipeline()
    print("Preprocessing regression data...")
    X_train_transformed = preprocessor.fit_transform(X_reg_train).toarray()
    X_val_transformed = preprocessor.transform(X_reg_val).toarray()
    X_test_transformed = preprocessor.transform(X_reg_test).toarray()

    results = []

    # Custom Linear Regression (no regularization)
    w, train_costs, val_costs = linear_regression_gradient_descent(
        X_train_transformed, y_reg_train.values, X_val_transformed, y_reg_val.values, reg_lambda=0.0
    )
    custom_reg_results = evaluate_linear_regression(
        X_train_transformed, y_reg_train.values, X_val_transformed, y_reg_val.values, 
        X_test_transformed, y_reg_test.values, w
    )
    cv_results = perform_cv("custom_linear", X_reg_train, y_reg_train, is_classification=False)
    print(f"Final Train MSE (no reg): {train_costs[-1]:.4f}, Val MSE (no reg): {val_costs[-1]:.4f}")
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
    print(f"Final Train MSE (L2): {train_costs_l2[-1]:.4f}, Val MSE (L2): {val_costs_l2[-1]:.4f}")
    results.append({
        "Model": "Custom Linear Regression (L2)",
        "CV MSE Mean": cv_results_l2['CV MSE Mean'],
        "CV MSE Std": cv_results_l2['CV MSE Std'],
        "MSE Train": custom_reg_results_l2["MSE Train"],
        "MSE Val": custom_reg_results_l2["MSE Val"],
        "MSE Test": custom_reg_results_l2["MSE Test"],
    })

    # Task 3.2: Experiment with PolynomialFeatures with scaling
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train_transformed)
    X_val_poly = poly.transform(X_val_transformed)
    X_test_poly = poly.transform(X_test_transformed)
    scaler_poly = StandardScaler()
    X_train_poly_scaled = scaler_poly.fit_transform(X_train_poly)
    X_val_poly_scaled = scaler_poly.transform(X_val_poly)
    X_test_poly_scaled = scaler_poly.transform(X_test_poly)
    w_poly, train_costs_poly, val_costs_poly = linear_regression_gradient_descent(
        X_train_poly_scaled, y_reg_train.values, X_val_poly_scaled, y_reg_val.values, reg_lambda=0.0
    )
    custom_reg_results_poly = evaluate_linear_regression(
        X_train_poly_scaled, y_reg_train.values, X_val_poly_scaled, y_reg_val.values, 
        X_test_poly_scaled, y_reg_test.values, w_poly
    )
    print(f"Final Train MSE (Poly): {train_costs_poly[-1]:.4f}, Val MSE (Poly): {val_costs_poly[-1]:.4f}")
    results.append({
        "Model": "Custom Linear Regression (Poly)",
        "CV MSE Mean": None,  # CV not performed for simplicity
        "CV MSE Std": None,
        "MSE Train": custom_reg_results_poly["MSE Train"],
        "MSE Val": custom_reg_results_poly["MSE Val"],
        "MSE Test": custom_reg_results_poly["MSE Test"],
    })

    # Convergence plots
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_costs)), train_costs, label='Train MSE (no reg)')
    plt.plot(range(len(val_costs)), val_costs, label='Val MSE (no reg)')
    plt.plot(range(len(train_costs_l2)), train_costs_l2, label='Train MSE (L2)', linestyle='--')
    plt.plot(range(len(val_costs_l2)), val_costs_l2, label='Val MSE (L2)', linestyle='--')
    plt.plot(range(len(train_costs_poly)), train_costs_poly, label='Train MSE (Poly)', linestyle='-.')
    plt.plot(range(len(val_costs_poly)), val_costs_poly, label='Val MSE (Poly)', linestyle='-.')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('Convergence Plot for Custom Linear Regression')
    plt.savefig('Part-2/results/convergence_plot.png')
    plt.close()
    print("Convergence plot saved.")

    # Sklearn models with adjusted Lasso
    for name, model in [
        ("SKLearn Linear Regression", LinearRegression()),
        ("Ridge Regression", Ridge(alpha=1.0)),
        ("Lasso Regression", Lasso(alpha=0.01, max_iter=10000)),  # Reduced alpha for better convergence
    ]:
        print(f"Evaluating {name}...")
        model.fit(X_train_transformed, y_reg_train)
        y_train_pred = model.predict(X_train_transformed)
        y_val_pred = model.predict(X_val_transformed)
        y_test_pred = model.predict(X_test_transformed)
        mse_train = mean_squared_error(y_reg_train, y_train_pred)
        mse_val = mean_squared_error(y_reg_val, y_val_pred)
        mse_test = mean_squared_error(y_reg_test, y_test_pred)
        cv_results = perform_cv(model, X_reg_train, y_reg_train, is_classification=False)
        results.append({
            "Model": name,
            "CV MSE Mean": cv_results['CV MSE Mean'],
            "CV MSE Std": cv_results['CV MSE Std'],
            "MSE Train": mse_train,
            "MSE Val": mse_val,
            "MSE Test": mse_test,
        })

    # Experiment with SelectKBest
    k = 10  # Select top 10 features
    selector = SelectKBest(score_func=f_regression, k=k)
    X_train_selected = selector.fit_transform(X_train_transformed, y_reg_train)
    X_val_selected = selector.transform(X_val_transformed)
    X_test_selected = selector.transform(X_test_transformed)
    for name, model in [
        ("SKLearn Linear Regression (Selected)", LinearRegression()),
        ("Ridge Regression (Selected)", Ridge(alpha=1.0)),
        ("Lasso Regression (Selected)", Lasso(alpha=0.01, max_iter=10000)),
    ]:
        print(f"Evaluating {name} with selected features...")
        model.fit(X_train_selected, y_reg_train)
        y_train_pred = model.predict(X_train_selected)
        y_val_pred = model.predict(X_val_selected)
        y_test_pred = model.predict(X_test_selected)
        mse_train = mean_squared_error(y_reg_train, y_train_pred)
        mse_val = mean_squared_error(y_reg_val, y_val_pred)
        mse_test = mean_squared_error(y_reg_test, y_test_pred)
        results.append({
            "Model": name,
            "CV MSE Mean": None,  # CV not performed for simplicity
            "CV MSE Std": None,
            "MSE Train": mse_train,
            "MSE Val": mse_val,
            "MSE Test": mse_test,
        })

    print("Regression evaluation completed.")
    return pd.DataFrame(results)