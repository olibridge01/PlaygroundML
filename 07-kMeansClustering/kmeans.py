import numpy as np

class KMeans:
    def __init__(self, k=3, max_iter=100, tol=1e-4):
        """
        Args:   
            k: number of clusters
            max_iter: maximum number of iterations
            tol: tolerance for convergence
        """
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self, X, verbose=False, initial_means=None):
        """
        Fit the k-means model to the data

        Args:
            X: (n, d) array of data
            verbose: if True, store the means and labels at each iteration
            initial_means: (k, d) array of initial means
        """

        # Initialise means
        self.means = X[np.random.choice(X.shape[0], self.k, replace=False)] if initial_means is None else initial_means
        self.labels = np.zeros(X.shape[0])
    
        if verbose:
            self.means_history = [self.means]
            self.labels_history = [self.labels]

        for t in range(self.max_iter):
            # Assign labels to each point
            self.labels = np.argmin(np.linalg.norm(X[:, None] - self.means, axis=2), axis=1)

            # Update means
            new_means = np.array([X[self.labels == i].mean(axis=0) for i in range(self.k)])

            # Check for convergence
            if np.linalg.norm(new_means - self.means) < self.tol:
                if verbose:
                    print(f' Converged after {t} iterations')
                break
            
            # Update means
            self.means = new_means

            # Append to history
            if verbose:
                self.means_history.append(self.means)
                self.labels_history.append(self.labels)
    
    def predict(self, X):
        """
        Predict the labels of test data

        Args:
            X: (n, d) array of data
        """
        return np.argmin(np.linalg.norm(X[:, None] - self.means, axis=2), axis=1)