from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Function to calculate and print evaluation metrics. These metrics include accuracy, precision, recall, F1 score, and confusion matrix
def evaluate_metrics(y_true, y_pred, model_name="", train_time=None, test_time=None):
    print(f"\n--- Evaluation for {model_name} ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average='macro', zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, average='macro', zero_division=0))
    print("F1 Score:", f1_score(y_true, y_pred, average='macro', zero_division=0))
    if train_time is not None:
        print(f"Training Time: {train_time:.4f} seconds")
    if test_time is not None:
        print(f"Testing Time: {test_time:.4f} seconds")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))