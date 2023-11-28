import numpy as np

def my_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Split the data into train and test sets.

    Args:
        X (np.array): The data to split.
        y (np.array): The labels to split.
        test_size (float): The percentage of data to use for testing.
        random_state (int): The seed to use for the random number generator.

    Returns:
        X_train (np.array): The training data.
        y_train (np.array): The training labels.
        X_test (np.array): The testing data.
        y_test (np.array): The testing labels.
    """
    rng = np.random.RandomState(random_state)
    indices = np.arange(X.shape[0])
    rng.shuffle(indices)
    split = int(X.shape[0] * test_size)
    X_train = X[indices[split:]]
    y_train = y[indices[split:]]

    X_test = X[indices[:split]]
    y_test = y[indices[:split]]

    return X_train, y_train, X_test, y_test


class myKNearestNeighbours:
    """
    A k-Nearest Neighbours classifier.
    """
    def __init__(self, k=5):
        """
        Initialise the classifier.

        Args:
            k (int): The number of neighbours to use.
        """
        self.k = k

    def fit(self, X, y):
        """
        Fit the classifier to the data.

        Args:
            X (np.array): The training data.
            y (np.array): The training labels.
        """
        self.X = X
        self.y = y

    def predict(self, X_test):
        """
        Predict the labels of the data.

        Args:
            X (np.array): The data to predict.

        Returns:
            y_pred (np.array): The predicted labels.
        """
        assert self.X is not None, "Classifier has not been fitted."
        assert X_test.shape[1] == self.X.shape[1], "Number of features in X_test does not match number of features in X."
        
        y_pred = np.zeros(X_test.shape[0])

        # Loop over each test point and find the k nearest neighbours.
        for i in range(X_test.shape[0]):
            distances = np.sqrt(np.sum((self.X - X_test[i])**2, axis=1))
            indices = np.argsort(distances)

            # Find the most common label of the k nearest neighbours.
            y_pred[i] = np.argmax(np.bincount(self.y[indices[:self.k]]))

        return y_pred

    def score(self, X_test, y_test):
        """
        Calculate the accuracy of the classifier.

        Args:
            X (np.array): The data to predict.
            y (np.array): The true labels.

        Returns:
            score (float): The accuracy of the classifier.
        """
        y_pred = self.predict(X_test)
        score = np.mean(y_pred == y_test)
        return score