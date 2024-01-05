import numpy as np

def rbf_kernel(sigma=1):
    """
    Compute the RBF kernel matrix.

    Args:
        X1 (np.array): First input data.
        X2 (np.array): Second input data.
        sigma (float): Kernel length scale.
    
    Returns:
        np.array: Kernel matrix.
    """
    return lambda X1, X2: np.exp(-(X1.reshape(-1, 1) - X2.reshape(1, -1))**2 / (2*sigma**2))

def linear_kernel():
    """
    Compute the linear kernel matrix.

    Args:
        X1 (np.array): First input data.
        X2 (np.array): Second input data.
    
    Returns:
        np.array: Kernel matrix.
    """
    return lambda X1, X2: X1 @ X2.T

class GaussianProcess:
    """Gaussian Process (GP) regression."""
    def __init__(self, mu, kernel, noise=0):
        self.mu = mu
        self.kernel = kernel
        self.noise = noise

    def get_prior(self, X, n_samples=1):
        """
        Sample from the GP prior

        Args:
            X (np.array): Input data.
            n_samples (int): Number of samples to draw.
        
        Returns:
            np.array: Samples from the prior.
            np.array: Standard deviation of the prior.
        """
        K = self.kernel(X, X)
        print(K.shape)
        f = np.random.multivariate_normal(self.mu(X).flatten(), K, size=n_samples).T
        std = np.sqrt(np.diag(K))
        return f, std
        
    def get_posterior(self, X_train, y_train, X_test):
        """
        Compute the posterior mean and covariance matrix.

        Args:
            X_train (np.array): Training data features.
            y_train (np.array): Training data labels.
            X_test (np.array): Test data features.
        
        Returns:
            np.array: Posterior mean.
            np.array: Posterior covariance matrix.
        """
        # Define kernel matrices
        K = self.kernel(X_train, X_train) + self.noise * np.eye(X_train.shape[0])
        Ks = self.kernel(X_train, X_test)
        Kss = self.kernel(X_test, X_test)

        # Compute posterior mean and covariance via Cholesky decomposition
        mean = Ks.T @ np.linalg.inv(K) @ y_train
        Cov = Kss - Ks.T @ np.linalg.inv(K) @ Ks

        return mean, Cov
    
    def posterior_sample(self, X_train, y_train, X_test, n_samples=1):
        """
        Sample from the posterior.

        Args:
            X_train (np.array): Training data features.
            y_train (np.array): Training data labels.
            X_test (np.array): Test data features.
            n_samples (int): Number of samples to draw.
        
        Returns:
            np.array: Posterior samples.
        """
        mean, Cov = self.get_posterior(X_train, y_train, X_test)
        return np.random.multivariate_normal(mean.flatten(), Cov, size=n_samples).T