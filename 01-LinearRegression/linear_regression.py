import numpy as np

class MyLinearRegression:
    """Linear regression model that uses the standard formula for the weights."""
    def __init__(self, linear_intercept=False):
        """
        Args:
            intercept (bool): Whether to include an intercept term.
        """
        self.linear_intercept = linear_intercept
        
    def fit(self, X_train, y_train, alpha=0):
        """
        Args:
            X_train (np.ndarray): Training data.
            y_train (np.ndarray): Training labels.
            alpha (float): Regularization parameter.
        """
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        if self.linear_intercept:
            X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)

        # Compute the weights using pseudoinverse of data matrix
        self.weights = np.linalg.inv(X_train.T @ X_train + alpha * np.eye(X_train.shape[1])) @ X_train.T @ y_train

    def predict(self, X_test):
        """
        Args:
            X_test (np.ndarray): Test data.
        Returns:
            np.ndarray: Predictions.
        """
        # Assert that the model has been fit
        assert hasattr(self, "weights"), "Model has not been fit yet."
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        if self.linear_intercept:
            X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)

        # Return predictions
        return X_test @ self.weights
    
    def mse(self, X_train, y_train):
        """
        Args:
            X_train (np.ndarray): Training data.
        Returns:
            float: Mean squared error.
        """
        # Assert that the model has been fit
        assert hasattr(self, "weights"), "Model has not been fit yet."
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        if self.linear_intercept:
            X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)

        # Return MSE
        return np.mean((X_train @ self.weights - y_train) ** 2)

