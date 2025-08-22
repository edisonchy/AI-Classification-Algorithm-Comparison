import numpy as np
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.classes = []
        self.class_probs = {}  # P(C)
        self.gaussian_params = {}  # (mean, variance) for continuous features

        # Train the model upon initialization
        self.fit()

    # Function to train the Naïve Bayes model
    def fit(self):
        # Extract unique class labels
        self.classes = np.unique(self.y_train)

        # Calculate P(C) (class probabilities) for each class
        class_counts = defaultdict(int)
        # Find the amount of times each class occurs
        for label in self.y_train:
            class_counts[label] += 1
        # The total number of samples
        total_samples = len(self.y_train)
        
        # Calculate P(C)
        self.class_probs = {cls: class_counts[cls] / total_samples for cls in self.classes}

        # Initialize storage for Gaussian parameters (mean, variance)
        self.gaussian_params = {cls: {} for cls in self.classes}

        # Calculate Gaussian parameters P(A | C) using mean and variance
        for cls in self.classes:
            cls_indices = np.where(self.y_train == cls)[0]
            cls_data = self.X_train[cls_indices]

            for j in range(self.X_train.shape[1]):  # Iterate over features
                feature_values = cls_data[:, j]
                mean = np.mean(feature_values)
                var = np.var(feature_values) if np.var(feature_values) > 1e-6 else 1e-6  # Ensure variance > 0
                self.gaussian_params[cls][j] = (mean, var)

    def _gaussian_prob(self, x, mean, var):
        """Compute Gaussian probability density function."""
        coeff = 1 / np.sqrt(2 * np.pi * var)
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return coeff * exponent

    def predict(self):
        """
        Predict class labels for self.X_test.

        :return: NumPy array of predicted class labels
        """
        predictions = []
        epsilon = 1e-9  # Small constant to avoid log(0)

        for x in self.X_test:
            class_scores = {}
            
            for cls in self.classes:
                # Start with P(C)
                score = np.log(self.class_probs[cls] + epsilon)  # Avoid log(0)

                for j in range(len(x)):
                    feature_value = x[j]
                    mean, var = self.gaussian_params[cls][j]
                    prob = self._gaussian_prob(feature_value, mean, var)
                    score += np.log(max(prob, epsilon))  # Clip probability to avoid log(0)

                class_scores[cls] = score

            # Choose class with highest probability
            predictions.append(max(class_scores, key=class_scores.get))

        return np.array(predictions)  # Convert to NumPy array for expected format
