# src/register_model.py
import shutil

def register_model():
    shutil.copy("model/model.pkl", "model/registered_model.pkl")
    print("Model registered (copied to model/registered_model.pkl)")

if __name__ == "__main__":
    register_model()
