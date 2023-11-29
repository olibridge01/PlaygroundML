import numpy as np
import matplotlib.pyplot as plt

class MyGaussianKernelRidgeRegression:
    def __init__(self, gamma, sigma):
        """
        Kernel ridge regression.
        
        Args:
            kernel (function): kernel function
            gamma (float): regularization parameter
        """
        self.gamma = gamma
        self.sigma = sigma

    def kernel(self, x_i, x_j):
        """
        Computes the Gaussian kernel between two vectors.
        
        Args:
            x_i (numpy.ndarray): vector
            x_j (numpy.ndarray): vector
            
        Returns:
            float: kernel value
        """
        return np.exp(-np.linalg.norm(x_i.T.reshape(x_i.shape[1],1,-1) - x_j.T.reshape(x_j.shape[1],-1,1), 
                                      axis=0)**2 / (2 * self.sigma**2))


    def fit(self, X_train, y_train):
        """
        Fit the model by computing the alpha values.
        
        Args:
            X_train (numpy.ndarray): training data
            y_train (numpy.ndarray): training output values
        """
        n_samples, n_features = X_train.shape
        K = self.kernel(X_train, X_train)
        self.alpha = np.linalg.inv(K + self.gamma * n_samples * np.eye(n_samples)) @ y_train

    def predict(self, X_train, X_test):
        """
        Predict y-values for new data.
        
        Args:
            X_train (numpy.ndarray): training data
            X_test (numpy.ndarray): test data

        Returns:
            numpy.ndarray: predicted y-values
        """
        K = self.kernel(X_train, X_test)
        return K @ self.alpha

    def mse(self, X_train, X_test, y_test):
        """
        Compute the mean squared error.
        
        Args:
            X_train (numpy.ndarray): training data
            X_test (numpy.ndarray): test data
            y_test (numpy.ndarray): test output values
            
        Returns:
            float: mean squared error
        """
        y_pred = self.predict(X_train, X_test)
        return np.mean((y_pred - y_test) ** 2)
    

def split_dataset(total_data, fraction):
    """
    Generates the training and test sets from total_data, given a fraction of data put into the training set.

    """
    shuffled_data = total_data
    np.random.shuffle(shuffled_data)

    # Split data into training and test sets
    train_set = shuffled_data[:int(np.round(fraction * len(shuffled_data)))]
    test_set = shuffled_data[int(np.round(fraction * len(shuffled_data))):]
    
    return train_set, test_set


def KRR_cross_validation(X_train, y_train, gammas, sigmas, k_validation):
    """
    Run k-fold cross-validation on the training set to determine (gamma,sigma) and perform KRR. Average MSEs over n_runs.

    Args:
        X_train (np.ndarray): training set
        y_train (np.ndarray): training output values
        gammas (np.ndarray): 1D array of gamma values
        sigmas (np.ndarray): 1D array of sigma values
        k_validation (int): number of folds for k-fold cross-validation

    Returns:
        i_best_gamma (int): index of best gamma
        i_best_sigma (int): index of best sigma
        parameter_MSEs (np.ndarray): 2D array of MSEs for each parameter pair (gamma,sigma)
    """

    # Split training set into k_validation folds
    X_kfold = np.array_split(X_train, k_validation)
    y_kfold = np.array_split(y_train, k_validation)

    # Initialize array to store MSEs for each parameter pair
    parameter_MSEs = np.zeros((len(gammas), len(sigmas)))

    # Iterate over each parameter pair
    for i_gamma, gamma in enumerate(gammas):
        print(f'Gamma {i_gamma + 1}/{len(gammas)} ', end='\r')
        for i_sigma, sigma in enumerate(sigmas):
            crossval_MSEs = []
            # Run k-fold cross-validation
            for i_fold in range(k_validation):
                X_crossval_test = X_kfold[i_fold]
                y_crossval_test = y_kfold[i_fold]

                X_crossval_train = np.concatenate([X_kfold[i] for i in range(k_validation) if i != i_fold])
                y_crossval_train = np.concatenate([y_kfold[i] for i in range(k_validation) if i != i_fold])
                
                KRR = MyGaussianKernelRidgeRegression(gamma, sigma)
                KRR.fit(X_crossval_train, y_crossval_train)

                crossval_MSE_test = KRR.mse(X_crossval_train, X_crossval_test, y_crossval_test)
                crossval_MSEs.append(crossval_MSE_test)
            
            mean_crossval_MSE = np.mean(crossval_MSEs)
            parameter_MSEs[i_gamma, i_sigma] = mean_crossval_MSE

    # Find best gamma and sigma
    i_best_gamma, i_best_sigma = np.unravel_index(np.argmin(parameter_MSEs), parameter_MSEs.shape)

    return i_best_gamma, i_best_sigma, parameter_MSEs


def plot_parameter_surface(parameter_MSEs, gammas, sigmas):
    """
    Plot the 3D surface for log(MSE) of each parameter pair (gamma,sigma).

    Args:
        parameter_MSEs (np.ndarray): 2D array of MSEs for each parameter pair (gamma,sigma).
        gammas (np.ndarray): 1D array of gamma values.
        sigmas (np.ndarray): 1D array of sigma values.
    """
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection="3d")
    loggammas, logsigmas = np.log(gammas) / np.log(2), np.log(sigmas) / np.log(2)

    X, Y = np.meshgrid(logsigmas, loggammas)
    Z = np.log(parameter_MSEs)

    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False)

    ax.set_zticks([2,4,6,8])
    ax.set_zlabel('$\log(MSE)$')

    ax.set_xticks(logsigmas[::4])
    ax.set_xticklabels(['$2^{{{}}}$'.format(int(x)) for x in logsigmas[::4]])

    ax.set_yticks(loggammas[::4])
    ax.set_yticklabels(['$2^{{{}}}$'.format(int(y)) for y in loggammas[::4]])

    # Shift figure to the left
    box = ax.get_position()
    ax.set_position([box.x0 - box.width * 0.15, box.y0, box.width * 0.85, box.height])

    ax.set_xlabel('$\sigma$')
    ax.set_ylabel('$\gamma$')
    ax.set_zlabel('$\ln$(MSE)')
    ax.set_box_aspect(aspect=None, zoom=0.8)
    plt.show()


def plot_contour(parameter_MSEs, gammas, sigmas):
    """
    Plot the contour plot for log(MSE) of each parameter pair (gamma,sigma).

    Args:
        parameter_MSEs (np.ndarray): 2D array of MSEs for each parameter pair (gamma,sigma).
        gammas (np.ndarray): 1D array of gamma values.
        sigmas (np.ndarray): 1D array of sigma values.
    """
    # Set axis ticks and labels
    levels = np.arange(2, 7, 0.25).tolist()
    loggammas, logsigmas = np.log(gammas)/np.log(2), np.log(sigmas)/np.log(2)

    # Plot contours
    fig, ax = plt.subplots(figsize=(3.6,3))
    contour = ax.contourf(logsigmas, loggammas, np.log(parameter_MSEs), levels=levels, cmap='viridis')
    fig.colorbar(contour, ax=ax, location='right', ticks=[2,3,4,5,6], label='$\ln$(MSE)')

    ax.set_xticks(logsigmas[::2])
    ax.set_xticklabels(['$2^{{{}}}$'.format(int(x)) for x in logsigmas[::2]])
    ax.set_yticks(loggammas[::2])
    ax.set_yticklabels(['$2^{{{}}}$'.format(int(y)) for y in loggammas[::2]])
    ax.set_xlabel('$\sigma$')
    ax.set_ylabel('$\gamma$')

    plt.show()