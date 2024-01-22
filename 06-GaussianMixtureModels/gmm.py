import numpy as np
from scipy.stats import multivariate_normal

class GaussianMixtureModel:
    def __init__(self, k, d):
        """
        Args:
            k: number of clusters
            d: dimensionality of data
        """
        self.k = k
        self.d = d
        self.means = np.zeros((k, d))
        self.covariances = np.array([np.eye(self.d)] * self.k)
        self.pi = np.random.rand(self.k)
        self.pi /= np.sum(self.pi)
        self.log_likelihood = 0

    def fit(self, X, max_iter=100):
        """
        Run EM algorithm to fit the model.

        Args:
            X: (n, d) array of data
            max_iter: maximum number of iterations
        """
        self.responsibilities = np.zeros((len(X), self.k))
        self.loglikes = []
        for i_iter in range(max_iter):
            # E-step
            self.E_step(X)

            # M-step
            self.M_step(X)

            # Append to list of log-likelihoods
            self.loglikes.append(self.log_likelihood)
    
    def fit_iterations(self, X, max_iter=100):
        """
        Run EM algorithm to fit the model and return the parameters at each iteration.

        Args:
            X: (n, d) array of data
            max_iter: maximum number of iterations
        """
        self.responsibilities = np.zeros((len(X), self.k))
        self.means_hist = []
        self.covariances_hist = []
        self.resp_hist = []
        self.loglikes = []
        for i_iter in range(max_iter):
            # E-step
            self.E_step(X)
            self.resp_hist.append(self.responsibilities.copy())

            # M-step
            self.M_step(X)
            self.means_hist.append(self.means.copy())
            self.covariances_hist.append(self.covariances.copy())

            # Append to list of log-likelihoods
            self.loglikes.append(self.log_likelihood)

    def E_step(self, X):
        """
        Run E-step of EM algorithm (compute responsibilities).

        Args:
            X: (n, d) array of data
        """
        for j in range(self.k):
            self.responsibilities[:, j] = self.pi[j] * self.gaussian(X, self.means[j], self.covariances[j])
        resp_sum = np.sum(self.responsibilities, axis=1).reshape(-1, 1)
        self.responsibilities /= resp_sum
        self.log_likelihood = np.sum(np.log(resp_sum))
    
    def M_step(self, X):
        """
        Run M-step of EM algorithm (update parameters).

        Args:
            X: (n, d) array of data
        """
        for j in range(self.k):
            sum_i = np.sum(self.responsibilities[:, j])
            self.means[j] = np.sum(self.responsibilities[:, j].reshape(-1, 1) * X, axis=0) / sum_i
            self.covariances[j] = np.sum(self.responsibilities[:, j].reshape(-1, 1, 1) 
                                         * np.array([np.outer(x - self.means[j], x - self.means[j]) for x in X]), axis=0) / sum_i
            self.pi[j] = sum_i / len(X)

    def gaussian(self, x, mean, covariance):
        """
        Return the pdf of a multivariate Gaussian distribution.

        Args:
            x: (d, ) array
            mean: (d, ) array
            covariance: (d, d) array

        Returns:
            float
        """
        return multivariate_normal.pdf(x, mean=mean, cov=covariance)