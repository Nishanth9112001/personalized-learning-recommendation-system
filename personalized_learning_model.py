"""
personalized_learning_model.py

Usage:
    python personalized_learning_model.py

This script:
 - loads the dataset 'personalized_learning_dataset.csv'
 - preprocesses features
 - trains a RandomForest classifier
 - prints evaluation metrics
 - saves a trained model to 'personalized_learning_model.pkl'
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os

DATA_PATH = "personalized_learning_dataset.csv"
MODEL_SAVE_PATH = "personalized_learning_model.pkl"

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    # Separate features and target
    X = df.drop(columns=['recommended_path','suggested_topic_focus'])
    y = df['recommended_path']
    # One-hot encode 'preferred_learning_style' (categorical numeric)
    cat_cols = ['preferred_learning_style','prior_knowledge_level']
    X_cat = X[cat_cols].astype(str)
    ohe = OneHotEncoder(sparse=False, drop='first')
    X_cat_enc = ohe.fit_transform(X_cat)
    cat_feature_names = ohe.get_feature_names_out(cat_cols)
    X_num = X.drop(columns=cat_cols).values
    X_processed = np.hstack([X_num, X_cat_enc])
    feature_names = list(X.drop(columns=cat_cols).columns) + list(cat_feature_names)
    return X_processed, y.values, feature_names, ohe

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, preds))
    return clf

def save_model(model, encoder, path=MODEL_SAVE_PATH):
    with open(path, 'wb') as f:
        pickle.dump({'model': model, 'encoder': encoder}, f)
    print(f"Saved trained model to {path}")

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Place the CSV in the same folder as this script.")
    df = load_data()
    X, y, feature_names, encoder = preprocess(df)
    print("Feature names used:", feature_names)
    model = train_and_evaluate(X, y)
    save_model(model, encoder)

if __name__ == "__main__":
    main()
