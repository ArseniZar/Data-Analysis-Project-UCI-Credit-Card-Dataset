import numpy as np
from custom_models import evaluate_logistic_regression, sigmoid

# Regresja logistyczna z regularyzacją L2
def logistic_regression_with_l2(X, y, learning_rate=0.01, epochs=100, batch_size=64, lambda_l2=0.1):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    n, p = X_b.shape
    w = np.zeros(p)
    class_weights = np.where(y == 1, len(y) / (2 * np.sum(y == 1)), len(y) / (2 * np.sum(y == 0)))
    
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        X_shuffled = X_b[indices]
        y_shuffled = y[indices]
        weights_shuffled = class_weights[indices]
        for i in range(0, n, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]
            w_batch = weights_shuffled[i:i + batch_size]
            z = X_batch @ w
            y_pred = sigmoid(z)
            grad = (X_batch.T @ ((y_pred - y_batch) * w_batch)) / X_batch.shape[0] + lambda_l2 * w
            w -= learning_rate * grad
    return w

if __name__ == "__main__":
    from pipeline import train_and_evaluate_models
    from data_loader import DataLoader
    
    loader = DataLoader("data/UCI_Credit_Card.csv")
    data = loader.data
    X_class = data.drop(columns=["ID", "default.payment.next.month"]).values
    y_class = data["default.payment.next.month"].values
    _, X_train, X_val, X_test, y_train, y_val, y_test = train_and_evaluate_models(X_class, y_class)
    
    # Bez regularyzacji
    w_no_reg = logistic_regression_with_l2(X_train, y_train, lambda_l2=0)
    metrics_no_reg = evaluate_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test, w_no_reg)
    
    # Z regularyzacją L2
    w_l2 = logistic_regression_with_l2(X_train, y_train, lambda_l2=0.1)
    metrics_l2 = evaluate_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test, w_l2)
    
    print("No regularization:", metrics_no_reg)
    print("L2 regularization:", metrics_l2)
    print("Weights (no reg):", w_no_reg[:5])
    print("Weights (L2):", w_l2[:5])