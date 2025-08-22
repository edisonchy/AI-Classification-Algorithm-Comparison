import numpy as np

class NaiveBayesClassifier:
    def __init__(self, epsilon=1e-9):
        self.classes = None
        self.class_priors = {}
        self.means = {}
        self.variances = {}
        # Set epsilon of small constant to avoid log(0)
        self.epsilon = epsilon

    # Train data by computing class priors, means, and variances
    def fit(self, X, y):
        # Convert X and y to numpy arrays
        X = np.array(X).astype(float)
        y = np.array(y)
        # Store classes
        self.classes, class_counts = np.unique(y, return_counts=True)
        n_samples, _ = X.shape
        #  Compute and store class priors
        self.class_priors = {cls: count / n_samples for cls, count in zip(self.classes, class_counts)}

        # Compute and store means and variances
        for cls in self.classes:
            X_c = X[y == cls]
            self.means[cls] = np.mean(X_c, axis=0)
            self.variances[cls] = np.var(X_c, axis=0) + self.epsilon

    # Function to compute log Gaussian likelihood
    def _log_gaussian_likelihood(self, x, mean, var):
        log_coeff = -0.5 * np.log(2 * np.pi * var)
        log_exp = -((x - mean) ** 2) / (2 * var)
        return log_coeff + log_exp

    # Function to predict class labels
    def predict(self, X):
        # Convert X to numpy array
        X = np.array(X).astype(float)

        # Initialize predictions
        predictions = []

        # Loop over each instance
        for inst in X:
            # Initialize class scores
            class_scores = {}

            # Compute log likelihood and log prior for each class
            for cls in self.classes:
                mean = self.means[cls]
                var = self.variances[cls]
                log_likelihoods = self._log_gaussian_likelihood(inst, mean, var)
                total_log_likelihood = np.sum(log_likelihoods)
                log_prior = np.log(self.class_priors[cls] + self.epsilon)
                total_score = log_prior + total_log_likelihood
                class_scores[cls] = total_score

            # Choose class with highest score
            max_score = max(class_scores.values())
            # Handle ties by selecting the class with the lowest index
            tied_classes = [cls for cls, score in class_scores.items() if score == max_score]
            predicted_class = min(tied_classes)

            predictions.append(predicted_class)

        return np.array(predictions)