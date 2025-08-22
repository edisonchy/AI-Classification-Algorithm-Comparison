import pandas as pd
import numpy as np
import time 
import algorithms.knn as knn
import algorithms.dt as dt
import algorithms.nb as nb
import algorithms.svm as svm
from sklearn.preprocessing import StandardScaler
from functions.functions import evaluate_metrics


# Loading dataset to pandas dataframe
column_names = ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"]
df = pd.read_csv("./data_file/glass.data", names=column_names)

# Shuffle with my student id
student_id = 39120767
np.random.seed(student_id)
indices = np.random.permutation(df.index)

# Split data into 80% training and 20% test sets
train_size = int(len(indices) * 0.8)
train_indices = indices[:train_size]
test_indices = indices[train_size:]
df_train = df.iloc[train_indices]
df_test = df.iloc[test_indices]

# Drop Id and Type columns, and split features (X) and labels (y) for training and testing sets
X_train = df_train.drop(columns=["Id", "Type"]).values
y_train = df_train["Type"].values

X_test = df_test.drop(columns=["Id", "Type"]).values
y_test = df_test["Type"].values

# Instantiate classifiers
knn_classifier = knn.KNNClassifier(X_train, y_train, X_test, y_test)
dt_classifier = dt.DecisionTreeClassifier(X_train, y_train, X_test, y_test)
nb_classifier = nb.NaiveBayesClassifier()
svm_classifier = svm.SupportVectorMachine()

# Normalize features by removing mean and scaling to unit variance to ensure equal contribution from each feature
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# || k-NN Comparison ||
start_time = time.time()
knn_predict = knn_classifier.predict()
test_time_knn = time.time() - start_time

evaluate_metrics(y_test, knn_predict, "KNN", train_time=0, test_time=test_time_knn)

# || Decision Tree Comparison ||
# Custom DecisionTreeClassifier
start_time = time.time()
dt_classifier.fit()
train_time_dt = time.time() - start_time

start_time = time.time()
dt_predict = dt_classifier.predict()
test_time_dt = time.time() - start_time

evaluate_metrics(y_test, dt_predict, "Decision Tree", train_time=train_time_dt, test_time=test_time_dt)

# || Naive Bayes Comparison ||
start_time = time.time()
nb_classifier.fit(X_train_scaled, y_train)
train_time_nb = time.time() - start_time

start_time = time.time()
nb_predict = nb_classifier.predict(X_test_scaled)
test_time_nb = time.time() - start_time

evaluate_metrics(y_test, nb_predict, "Naive Bayes", train_time=train_time_nb, test_time=test_time_nb)

# --- SVM Comparison ---
start_time = time.time()
svm_classifier.fit(X_train_scaled, y_train) 
train_time_svm = time.time() - start_time

start_time = time.time()
svm_predict = svm_classifier.predict(X_test_scaled)
test_time_svm = time.time() - start_time

evaluate_metrics(y_test, svm_predict, "Support Vector Machine", train_time=train_time_svm, test_time=test_time_svm)

