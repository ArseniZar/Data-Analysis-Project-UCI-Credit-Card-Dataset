from imblearn.over_sampling import SMOTE
from custom_models import logistic_regression_gradient_descent, evaluate_logistic_regression
from sklearn.metrics import classification_report

if __name__ == "__main__":
    from pipeline import train_and_evaluate_models
    from data_loader import DataLoader
    
    loader = DataLoader("data/UCI_Credit_Card.csv")
    data = loader.data
    X_class = data.drop(columns=["ID", "default.payment.next.month"]).values
    y_class = data["default.payment.next.month"].values
    _, X_train, X_val, X_test, y_train, y_val, y_test = train_and_evaluate_models(X_class, y_class)
    
    # Bez balansowania
    w_no_balance = logistic_regression_gradient_descent(X_train, y_train)
    metrics_no_balance = evaluate_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test, w_no_balance)
    
    # Z SMOTE
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    w_bal = logistic_regression_gradient_descent(X_train_bal, y_train_bal)
    metrics_bal = evaluate_logistic_regression(X_train_bal, y_train_bal, X_val, y_val, X_test, y_test, w_bal)
    
    print("No balancing:", metrics_no_balance)
    print("With SMOTE:", metrics_bal)