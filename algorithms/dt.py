import numpy as np
from collections import Counter


class DecisionTreeClassifier:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)
        self.tree = None

    # Function for entropy calculation
    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    # Function for information gain
    def _information_gain(self, X, y, feature_idx, threshold):
        total_entropy = self._entropy(y)

        # Split based on threshold (X[:, feature_idx] <= threshold)
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        # Handle empty splits
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0  

        # Compute for entortropy for left and right branches
        left_entropy = self._entropy(y[left_mask])
        right_entropy = self._entropy(y[right_mask])

        # Compute weighted entropy
        weighted_entropy = (np.sum(left_mask) / len(y)) * left_entropy + (np.sum(right_mask) / len(y)) * right_entropy

        return total_entropy - weighted_entropy

    # Function to compute best split
    def _best_split(self, X, y):
        best_gain = -1
        best_threshold = None
        best_feature = None

        # Loop over all features to find the best split
        for feature_idx in range(X.shape[1]):
            # Get unique values from current feature
            values = np.unique(X[:, feature_idx])
            # Consider thresholds between consecutive values
            thresholds = (values[:-1] + values[1:]) / 2

            # Loop over all possible thresholds
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature_idx, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    # Function to recursively build the decision tree
    def _build_tree(self, X, y):
        # if all labels are the same, return that label
        if len(np.unique(y)) == 1:
            return y[0]
        
        # if no more features, return the most common label
        if X.shape[1] == 0:
            return Counter(y).most_common(1)[0][0]

        # Find the best split, if no split is found, return the most common label
        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return Counter(y).most_common(1)[0][0]

        # Initialise the tree
        tree = {
            best_feature: {"threshold": best_threshold, "left": None, "right": None}
        }

        # Split data based on threshold and create left and right subtrees
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        sub_X_left = X[left_mask]
        sub_y_left = y[left_mask]
        sub_X_right = X[right_mask]
        sub_y_right = y[right_mask]

        # Recursively build left and right subtrees
        tree[best_feature]["left"] = self._build_tree(sub_X_left, sub_y_left)
        tree[best_feature]["right"] = self._build_tree(sub_X_right, sub_y_right)

        return tree

    # Function to train the decision tree
    def fit(self):
        self.tree = self._build_tree(self.X_train, self.y_train)

    # Function to predict the class of a single instance
    def _predict_instance(self, instance, tree):
        # If tree is a leaf node, return its label
        if not isinstance(tree, dict):
            return tree

        # Otherwise, recursively traverse the tree
        feature_idx = list(tree.keys())[0]
        threshold = tree[feature_idx]["threshold"]

        if instance[feature_idx] <= threshold:
            return self._predict_instance(instance, tree[feature_idx]["left"])
        else:
            return self._predict_instance(instance, tree[feature_idx]["right"])

    # Function to predict the class of all instances
    def predict(self):
        # Check if the tree has been trained
        if self.tree is None:
            raise ValueError("The tree has not been trained. Call fit() first.")

        # Predict the class of all instances
        predictions = [
            self._predict_instance(instance, self.tree) for instance in self.X_test
        ]
        return np.array(predictions)
