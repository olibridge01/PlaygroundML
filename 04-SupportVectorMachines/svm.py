import numpy as np
from scipy.optimize import minimize

class SupportVectorMachine:
    """Support Vector Machine (SVM) classifier."""
    def __init__(self, C=1.0):
        """
        Initialise SVM with hyperparameter C.

        Args: 
            C (float): Regularisation parameter for soft margin SVM.
        """
        self.w = None
        self.b = None
        self.C = C

    def fit(self, X, y):
        """
        Fit SVM to training data.

        Args:
            X (np.array): Training data features.
            y (np.array): Training data labels.
        
        Returns:
            bool: True if optimisation successful, False otherwise.
        """
        n_points, _ = X.shape
        y = y.reshape(-1, 1) * 1.
        A = (y * X) @ (y * X).T

        def objective_func(alpha):
            """Dual optimisation objective."""
            return 0.5 * alpha.T @ A @ alpha - np.sum(alpha)

        def constraint_func(alpha):
            """Constraint function."""
            return alpha @ y

        # Optimisation constraints
        bounds = [(0, self.C) for _ in range(n_points)]
        constraint = {'type': 'eq', 'fun': lambda a: constraint_func(a)}

        # Solve minimisation problem subject to constraints
        res = minimize(lambda a: objective_func(a), np.zeros(n_points), bounds=bounds, constraints=constraint)
        if res['success']:
            alpha = res.x

            # Compute weight vector from dual solution
            self.w = np.sum((alpha.reshape(-1, 1) * y) * X, axis=0)

            # Compute b from KKT conditions (for support vectors)
            S = (alpha > 1e-5).flatten()
            self.b = np.mean(y[S] - X[S] @ self.w)

            # Store support vectors
            self.support_vectors = X[S]
            self.support_vector_labels = y[S]

            return True
        else:
            return False