import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import time

# ===============================
# Logistic Regression (NumPy)
# ===============================

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def logistic_regression_gradient_descent(X: np.ndarray, y: np.ndarray, learning_rate: float = 0.01, 
                                        epochs: int = 100, batch_size: int = 64, reg_lambda: float = 0.0) -> np.ndarray:
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
            if reg_lambda > 0:  # Добавляем L2 регуляризацию
                grad[1:] += reg_lambda * w[1:]  # Исключаем bias
            w -= learning_rate * grad
    return w

def predict_logistic_regression(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return (sigmoid(X_b @ w) > 0.5).astype(int)

def evaluate_logistic_regression(X_train: np.ndarray, y_train: np.ndarray, 
                                X_val: np.ndarray, y_val: np.ndarray, 
                                X_test: np.ndarray, y_test: np.ndarray, 
                                w: np.ndarray) -> dict:
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

# ===============================
# Linear Regression (NumPy)
# ===============================

def linear_regression_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    w = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
    return w

def linear_regression_gradient_descent(X_train: np.ndarray, y_train: np.ndarray, 
                                       X_val: np.ndarray, y_val: np.ndarray, 
                                       learning_rate: float = 0.01, epochs: int = 100, 
                                       batch_size: int = 64, reg_lambda: float = 0.0):
    X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_val_b = np.c_[np.ones((X_val.shape[0], 1)), X_val]
    n, p = X_train_b.shape
    w = np.zeros(p)
    train_costs = []
    val_costs = []
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        X_shuffled = X_train_b[indices]
        y_shuffled = y_train[indices]
        for i in range(0, n, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]
            y_pred = X_batch @ w
            grad = (X_batch.T @ (y_pred - y_batch)) / X_batch.shape[0]
            if reg_lambda > 0:  # Добавляем L2 регуляризацию
                grad[1:] += reg_lambda * w[1:]  # Исключаем bias
            w -= learning_rate * grad
        # Считаем стоимость для train и val
        y_train_pred = X_train_b @ w
        train_cost = np.mean((y_train - y_train_pred) ** 2)
        train_costs.append(train_cost)
        y_val_pred = X_val_b @ w
        val_cost = np.mean((y_val - y_val_pred) ** 2)
        val_costs.append(val_cost)
    return w, train_costs, val_costs

def predict_linear_regression(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return X_b @ w

def evaluate_linear_regression(X_train: np.ndarray, y_train: np.ndarray, 
                              X_val: np.ndarray, y_val: np.ndarray, 
                              X_test: np.ndarray, y_test: np.ndarray, 
                              w: np.ndarray) -> dict:
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

# ===============================
# Logistic Regression (PyTorch)
# ===============================

class LogisticRegressionPyTorch(nn.Module):
    def __init__(self, input_dim: int):
        super(LogisticRegressionPyTorch, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))

def train_pytorch_model(X_train: np.ndarray, y_train: np.ndarray, 
                       X_val: np.ndarray, y_val: np.ndarray, 
                       X_test: np.ndarray, y_test: np.ndarray, 
                       device: str = 'cpu', batch_size: int = 64, 
                       num_epochs: int = 100, lr: float = 0.01) -> dict:
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size)
    model = LogisticRegressionPyTorch(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    training_time = time.time() - start_time
    def evaluate(loader):
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                predictions = (outputs > 0.5).float()
                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred)
    train_acc, train_f1 = evaluate(train_loader)
    val_acc, val_f1 = evaluate(val_loader)
    test_acc, test_f1 = evaluate(test_loader)
    return {
        'model': model,
        'training_time': training_time,
        'Accuracy Train': train_acc,
        'F1 Train': train_f1,
        'Accuracy Val': val_acc,
        'F1 Val': val_f1,
        'Accuracy Test': test_acc,
        'F1 Test': test_f1
    }