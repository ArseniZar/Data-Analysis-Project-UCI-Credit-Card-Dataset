import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from custom_models import predict_logistic_regression, sigmoid

# Zmodyfikowana funkcja z zapisywaniem kosztu
def logistic_regression_with_cost(X, y, learning_rate=0.01, epochs=100, batch_size=64):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    n, p = X_b.shape
    w = np.zeros(p)
    class_weights = np.where(y == 1, len(y) / (2 * np.sum(y == 1)), len(y) / (2 * np.sum(y == 0)))
    train_costs, val_costs = [], []
    
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
            grad = (X_batch.T @ ((y_pred - y_batch) * w_batch)) / X_batch.shape[0]
            w -= learning_rate * grad
        
        # Oblicz koszt dla zbioru treningowego i walidacyjnego
        y_train_pred = sigmoid(X_b @ w)
        train_cost = -np.mean(y * np.log(y_train_pred + 1e-10) + (1 - y) * np.log(1 - y_train_pred + 1e-10))
        train_costs.append(train_cost)
    
    return w, train_costs

# Analiza zbieżności
def analyze_convergence(X_train, y_train, X_val, y_val):
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)
    
    w, train_costs = logistic_regression_with_cost(X_train_poly, y_train)
    val_costs = []
    for epoch in range(len(train_costs)):
        y_val_pred = predict_logistic_regression(X_val_poly, w)
        val_cost = -np.mean(y_val * np.log(y_val_pred + 1e-10) + (1 - y_val) * np.log(1 - y_val_pred + 1e-10))
        val_costs.append(val_cost)
    
    plt.plot(train_costs, label="Train Cost")
    plt.plot(val_costs, label="Validation Cost")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.legend()
    plt.title("Convergence Analysis")
    plt.show()

if __name__ == "__main__":
    from pipeline import train_and_evaluate_models
    from data_loader import DataLoader
    
    loader = DataLoader("data/UCI_Credit_Card.csv")
    data = loader.data
    X_class = data.drop(columns=["ID", "default.payment.next.month"]).values
    y_class = data["default.payment.next.month"].values
    _, X_train, X_val, _, y_train, y_val, _ = train_and_evaluate_models(X_class, y_class)
    analyze_convergence(X_train, y_train, X_val, y_val)