import numpy as np

class SupportVectorMachine:
    def __init__(self, C=1.0, Tolerance=1e-3, max_iter=100):
        # Regularization parameter C controls the trade-off between margin maximization and classification error
        self.C = C
        # Tolerance for stopping criteria in SMO (how close we need to be to the optimal solution)
        self.tol = Tolerance
        # Maximum number of iterations for the SMO algorithm to prevent infinite loops
        self.max_iter = max_iter
        # Dictionary to store binary SVM models for each pair of classes (one-vs-one strategy)
        self.models = {}
        # Array to store the unique class labels in the dataset
        self.classes = None

    # Simple linear kernel: just the dot product of two vectors
    def _linear_kernel(self, vec1, vec2):
        return np.dot(vec1, vec2)

    # Train a binary SVM for two classes (class_a vs. class_b) using the SMO algorithm
    def _fit_binary(self, data, labels, class_a, class_b):
        # Filter the data to only include samples from the two classes we're training on
        mask = (labels == class_a) | (labels == class_b)
        X_subset = data[mask]
        y_subset = labels[mask]
        # Convert labels to +1/-1 for binary classification (class_a -> +1, class_b -> -1)
        y_subset = np.where(y_subset == class_a, 1, -1)
        num_samples, num_features = X_subset.shape

        # Initialize Lagrange multipliers (alphas), bias (b), and weight vector (w)
        alphas = np.zeros(num_samples)
        b = 0.0
        w = np.zeros(num_features)

        # Precompute the kernel matrix for all pairs of samples (for linear kernel, it's just dot products)
        kernel_matrix = np.zeros((num_samples, num_samples))
        for i in range(num_samples):
            for j in range(num_samples):
                kernel_matrix[i, j] = self._linear_kernel(X_subset[i], X_subset[j])

        # Start the SMO algorithm to optimize alphas
        iterations = 0
        while iterations < self.max_iter:
            updates = 0  # Track how many alpha updates we make in this iteration
            for i in range(num_samples):
                # Compute the error for sample i: f(x_i) - y_i, where f(x_i) = sum(alpha_j * y_j * K(x_i, x_j)) + b
                error_i = np.dot(alphas * y_subset, kernel_matrix[i]) + b - y_subset[i]

                # Check if sample i violates the KKT conditions (i.e., needs optimization)
                if ((y_subset[i] * error_i < -self.tol and alphas[i] < self.C) or
                    (y_subset[i] * error_i > self.tol and alphas[i] > 0)):

                    # Pick a second sample j randomly, but make sure it's not the same as i
                    j = np.random.randint(0, num_samples)
                    while j == i:
                        j = np.random.randint(0, num_samples)

                    # Compute the error for sample j
                    error_j = np.dot(alphas * y_subset, kernel_matrix[j]) + b - y_subset[j]

                    # Store the old values of alpha_i and alpha_j before updating
                    old_alpha_i = alphas[i]
                    old_alpha_j = alphas[j]

                    # Compute the bounds L and H for alpha_j based on the constraints
                    if y_subset[i] != y_subset[j]:
                        low = max(0, alphas[j] - alphas[i])
                        high = min(self.C, self.C + alphas[j] - alphas[i])
                    else:
                        low = max(0, alphas[i] + alphas[j] - self.C)
                        high = min(self.C, alphas[i] + alphas[j])

                    # If L == H, skip to the next sample because no update is possible
                    if low == high:
                        continue

                    # Compute eta, which is used to update alpha_j (eta = 2K_ij - K_ii - K_jj)
                    eta = 2 * kernel_matrix[i, j] - kernel_matrix[i, i] - kernel_matrix[j, j]
                    if eta >= 0:  # If eta >= 0, the update won't improve the objective, so skip
                        continue

                    # Update alpha_j using the SMO update rule
                    alphas[j] = alphas[j] - y_subset[j] * (error_i - error_j) / eta
                    alphas[j] = max(low, min(high, alphas[j]))  # Clip alpha_j to stay within bounds

                    # Update alpha_i based on the change in alpha_j (maintains the constraint sum(alpha_i * y_i) = 0)
                    alphas[i] = alphas[i] + y_subset[i] * y_subset[j] * (old_alpha_j - alphas[j])

                    # Update the bias b using the new alphas
                    b1 = b - error_i - y_subset[i] * (alphas[i] - old_alpha_i) * kernel_matrix[i, i] \
                         - y_subset[j] * (alphas[j] - old_alpha_j) * kernel_matrix[i, j]
                    b2 = b - error_j - y_subset[i] * (alphas[i] - old_alpha_i) * kernel_matrix[i, j] \
                         - y_subset[j] * (alphas[j] - old_alpha_j) * kernel_matrix[j, j]

                    # Choose the new bias based on which alpha is within (0, C)
                    if 0 < alphas[i] < self.C:
                        b = b1
                    elif 0 < alphas[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0

                    updates += 1  # Increment the update counter

            iterations += 1
            if updates == 0:  # If no updates were made, we've converged, so stop
                break

        # Compute the weight vector w = sum(alpha_i * y_i * x_i) for the linear kernel
        w = np.dot((alphas * y_subset).reshape(-1, 1).T, X_subset).flatten()

        # Identify support vectors (samples where alpha > 0)
        support_idx = alphas > 1e-5
        supports = X_subset[support_idx]
        support_y = y_subset[support_idx]
        support_alpha = alphas[support_idx]

        # Return the trained parameters and support vectors
        return w, b, supports, support_y, support_alpha

    # Train the SVM using a one-vs-one strategy for multiclass classification
    def fit(self, X_train, y_train):
        # Convert inputs to numpy arrays and ensure float type
        X_train = np.array(X_train, dtype=float)
        y_train = np.array(y_train)
        # Get the unique classes in the dataset
        self.classes = np.unique(y_train)

        # Train a binary SVM for each pair of classes
        for i, c1 in enumerate(self.classes):
            for c2 in self.classes[i + 1:]:
                # Train the binary SVM and store the model
                trained_model = self._fit_binary(X_train, y_train, c1, c2)
                self.models[(c1, c2)] = trained_model

    # Predict using a binary SVM model (returns +1 or -1)
    def _predict_binary(self, X_test, weights, bias):
        # Compute the decision function w * x + b and return the sign
        return np.sign(np.dot(X_test, weights) + bias)

    # Predict using one-vs-one voting for multiclass classification
    def predict(self, X_test):
        # Convert test data to numpy array and ensure float type
        X_test = np.array(X_test, dtype=float)
        num_tests = X_test.shape[0]
        # Initialize a vote matrix: each row is a test sample, each column is a class
        votes = np.zeros((num_tests, len(self.classes)))

        # For each pair of classes, use the binary SVM to predict and cast votes
        for (cls1, cls2), (w, b, _, _, _) in self.models.items():
            predictions = self._predict_binary(X_test, w, b)
            # Get the indices of cls1 and cls2 in the classes array
            idx1 = np.where(self.classes == cls1)[0][0]
            idx2 = np.where(self.classes == cls2)[0][0]

            # For each test sample, add a vote for the predicted class
            for sample in range(num_tests):
                if predictions[sample] == 1:
                    votes[sample, idx1] += 1
                else:
                    votes[sample, idx2] += 1

        # Predict the class with the most votes for each test sample
        return self.classes[np.argmax(votes, axis=1)]