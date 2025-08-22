import numpy as np

class SupportVectorMachine:
    def __init__(self, C=1.0, Tolerance=1e-3, max_iter=100):
        self.C = C
        self.tol = Tolerance
        self.max_iter = max_iter
        self.models = {}
        self.classes = None

    # Compute for linear kernel between two data points
    def _linear_kernel(self, vec1, vec2):
        return np.dot(vec1, vec2)

    # Train a binary SVM for class1 vs. class2 using SMO
    def _fit_binary(self, data, labels, class_a, class_b):
        mask = (labels == class_a) | (labels == class_b)
        X_subset = data[mask]
        y_subset = labels[mask]
        y_subset = np.where(y_subset == class_a, 1, -1)
        num_samples, num_features = X_subset.shape

        alphas = np.zeros(num_samples)
        b = 0.0
        w = np.zeros(num_features)

        kernel_matrix = np.zeros((num_samples, num_samples))
        for i in range(num_samples):
            for j in range(num_samples):
                kernel_matrix[i, j] = self._linear_kernel(X_subset[i], X_subset[j])

        iterations = 0
        while iterations < self.max_iter:
            updates = 0
            for i in range(num_samples):
                error_i = np.dot(alphas * y_subset, kernel_matrix[i]) + b - y_subset[i]

                if ((y_subset[i] * error_i < -self.tol and alphas[i] < self.C) or
                    (y_subset[i] * error_i > self.tol and alphas[i] > 0)):

                    j = np.random.randint(0, num_samples)
                    while j == i:  # Make sure j isn’t i
                        j = np.random.randint(0, num_samples)

                    error_j = np.dot(alphas * y_subset, kernel_matrix[j]) + b - y_subset[j]

                    old_alpha_i = alphas[i]
                    old_alpha_j = alphas[j]

                    if y_subset[i] != y_subset[j]:
                        low = max(0, alphas[j] - alphas[i])
                        high = min(self.C, self.C + alphas[j] - alphas[i])
                    else:
                        low = max(0, alphas[i] + alphas[j] - self.C)
                        high = min(self.C, alphas[i] + alphas[j])

                    if low == high:
                        continue

                    eta = 2 * kernel_matrix[i, j] - kernel_matrix[i, i] - kernel_matrix[j, j]
                    if eta >= 0:
                        continue

                    alphas[j] = alphas[j] - y_subset[j] * (error_i - error_j) / eta
                    alphas[j] = max(low, min(high, alphas[j]))

                    alphas[i] = alphas[i] + y_subset[i] * y_subset[j] * (old_alpha_j - alphas[j])

                    b1 = b - error_i - y_subset[i] * (alphas[i] - old_alpha_i) * kernel_matrix[i, i] \
                         - y_subset[j] * (alphas[j] - old_alpha_j) * kernel_matrix[i, j]
                    b2 = b - error_j - y_subset[i] * (alphas[i] - old_alpha_i) * kernel_matrix[i, j] \
                         - y_subset[j] * (alphas[j] - old_alpha_j) * kernel_matrix[j, j]

                    if 0 < alphas[i] < self.C:
                        b = b1
                    elif 0 < alphas[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0

                    updates += 1

            iterations += 1
            if updates == 0:
                break

        w = np.dot((alphas * y_subset).reshape(-1, 1).T, X_subset).flatten()

        support_idx = alphas > 1e-5
        supports = X_subset[support_idx]
        support_y = y_subset[support_idx]
        support_alpha = alphas[support_idx]

        return w, b, supports, support_y, support_alpha

    # Train a one vs one classifier for each pair of classes
    def fit(self, X_train, y_train):
        X_train = np.array(X_train, dtype=float)
        y_train = np.array(y_train)
        self.classes = np.unique(y_train)

        for i, c1 in enumerate(self.classes):
            for c2 in self.classes[i + 1:]:
                trained_model = self._fit_binary(X_train, y_train, c1, c2)
                self.models[(c1, c2)] = trained_model

    # Predict using a binary SVM 
    def _predict_binary(self, X_test, weights, bias):
        return np.sign(np.dot(X_test, weights) + bias)

    # Predict using one-vs-one voting
    def predict(self, X_test):
        X_test = np.array(X_test, dtype=float)
        num_tests = X_test.shape[0]
        votes = np.zeros((num_tests, len(self.classes)))

        for (cls1, cls2), (w, b, _, _, _) in self.models.items():
            predictions = self._predict_binary(X_test, w, b)
            idx1 = np.where(self.classes == cls1)[0][0]
            idx2 = np.where(self.classes == cls2)[0][0]

            for sample in range(num_tests):
                if predictions[sample] == 1:
                    votes[sample, idx1] += 1
                else:
                    votes[sample, idx2] += 1

        return self.classes[np.argmax(votes, axis=1)]