import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def binary_cross_entropy(y_true, y_pred, weights=None):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    if weights is not None:
        loss = loss * weights
    return np.mean(loss)

def logistic_regression_gradient_descent(X, y, learning_rate=0.01, epochs=100, batch_size=64):
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
            grad = (X_batch.T @ ((y_pred - y_batch) * w_batch)) / X_batch.shape[0]
            w -= learning_rate * grad
    return w

def predict_logistic_regression(X, w):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return (sigmoid(X_b @ w) > 0.5).astype(int)

def evaluate_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test, w):
    y_train_pred = predict_logistic_regression(X_train, w)
    y_val_pred = predict_logistic_regression(X_val, w)
    y_test_pred = predict_logistic_regression(X_test, w)
    return {
        'Accuracy Train': accuracy_score(y_train, y_train_pred),
        'F1 Train': f1_score(y_train, y_train_pred),
        'Accuracy Val': accuracy_score(y_val, y_val_pred),
        'F1 Val': f1_score(y_val, y_val_pred),
        'Accuracy Test': accuracy_score(y_test, y_test_pred),
        'F1 Test': f1_score(y_test, y_test_pred)
    }

def linear_regression_closed_form(X, y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    w = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
    return w

def predict_linear_regression(X, w):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return X_b @ w

def evaluate_linear_regression(X_train, y_train, X_val, y_val, X_test, y_test, w):
    y_train_pred = predict_linear_regression(X_train, w)
    y_val_pred = predict_linear_regression(X_val, w)
    y_test_pred = predict_linear_regression(X_test, w)
    mse_train = np.mean((y_train - y_train_pred) ** 2)
    mse_val = np.mean((y_val - y_val_pred) ** 2)
    mse_test = np.mean((y_test - y_test_pred) ** 2)
    return {
        'MSE Train': mse_train,
        'MSE Val': mse_val,
        'MSE Test': mse_test
    }
