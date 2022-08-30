import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

def eval_multi_model(model, X, y, dataset_name):
    # Perform stratified train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123, stratify=y)

    # Fit model
    model.fit(X_train, y_train)
    # Predict test labels
    y_pred = model.predict(X_test)
    
    # Evaluate performance
    print('Dataset: ' + dataset_name)
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print(f'Balanced accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}')
    print()
    
def eval_bin_model(model, X, y, dataset_name):
    # Perform stratified train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123, stratify=y)

    # Fit model
    model.fit(X_train, y_train)
    # Predict test labels
    y_pred = model.predict(X_test)
    # Evaluate performance
    print('Dataset: ' + dataset_name)
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print(f'Balanced accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}')
    print(f'AUC: {roc_auc_score(y_test.map({"happy": 1, "not_happy": 0}).values, [1 if p == "happy" else 0 for p in y_pred]):.4f}')
    print()