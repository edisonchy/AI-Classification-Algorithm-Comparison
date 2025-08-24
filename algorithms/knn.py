import numpy as np
from collections import Counter


class KNNClassifier:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)

        # Compute k using only training data to avoid peeking at the test set
        total_data_points_num = len(X_train)
        k = int(np.sqrt(total_data_points_num))

        # Ensure k is odd and does not exceed the number of training samples
        if k % 2 == 0:
            k = k + 1 if k + 1 <= total_data_points_num else k - 1
        self.k = k

    # Euclidean distance of query point and each training sample
    def _euclidean_distance(self, query_point, train_sample):
        return np.sqrt(np.sum((query_point - train_sample) ** 2))

    # Predict the class label for each test sample
    def predict(self):
        predictions = []
        # For each test sample in X_test
        for test_sample in self.X_test:
            # Compute distances between for that particular test sameple and all training samples
            distances = [
                self._euclidean_distance(test_sample, train_sample)
                for train_sample in self.X_train
            ]

            # Get the indices of k nearest neighbors in sorted distances
            k_indices = np.argsort(distances)[: self.k]

            # Get the labels of k nearest neighbors
            k_nearest_labels = [self.y_train[i] for i in k_indices]

            # Determine the most common class label
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]

            # Append the predicted class label to predictions list
            predictions.append(most_common)

        return np.array(predictions)
