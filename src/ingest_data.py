# src/ingest_data.py
import pandas as pd
from sklearn.datasets import load_iris

def ingest_data():
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.to_csv("data/iris.csv", index=False)
    print("Data saved to data/iris.csv")

if __name__ == "__main__":
    ingest_data()
