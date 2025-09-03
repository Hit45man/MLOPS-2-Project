# src/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_and_tune():
    df = pd.read_csv("data/iris.csv")
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        "n_estimators": [10, 50, 100],
        "max_depth": [None, 10, 20]
    }

    model = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
    model.fit(X_train, y_train)

    os.makedirs("model", exist_ok=True)
    joblib.dump(model.best_estimator_, "model/model.pkl")
    print("Best model saved to model/model.pkl")

if __name__ == "__main__":
    train_and_tune()
