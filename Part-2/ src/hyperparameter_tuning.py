from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from custom_models import logistic_regression_gradient_descent, predict_logistic_regression

class CustomLogisticWrapper:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def fit(self, X, y):
        self.w = logistic_regression_gradient_descent(X, y, self.learning_rate, self.epochs)
        return self
    
    def predict(self, X):
        return predict_logistic_regression(X, self.w)

if __name__ == "__main__":
    from pipeline import train_and_evaluate_models
    from data_loader import DataLoader
    
    loader = DataLoader("data/UCI_Credit_Card.csv")
    data = loader.data
    X_class = data.drop(columns=["ID", "default.payment.next.month"]).values
    y_class = data["default.payment.next.month"].values
    _, X_train, X_val, X_test, y_train, y_val, y_test = train_and_evaluate_models(X_class, y_class)
    
    # GridSearch dla sklearn LogisticRegression
    param_grid_sk = {'C': [0.1, 1, 10]}
    grid_sk = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_sk, cv=3)
    grid_sk.fit(X_train, y_train)
    print("Best params (sklearn):", grid_sk.best_params_)
    
    # GridSearch dla Custom Logistic
    param_grid_custom = {'learning_rate': [0.001, 0.01, 0.1], 'epochs': [50, 100, 200]}
    grid_custom = GridSearchCV(CustomLogisticWrapper(), param_grid_custom, cv=3)
    grid_custom.fit(X_train, y_train)
    print("Best params (custom):", grid_custom.best_params_)