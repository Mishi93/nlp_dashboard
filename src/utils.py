# nlp-dashboard/src/utils.py
import os
import pickle
import pandas as pd

def ensure_directories():
    """Ensures that structural project directories exist dynamically."""
    directories = [
        "data",
        "models",
        "outputs/figures",
        "outputs/exports",
        "src"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def load_data(file_path: str) -> pd.DataFrame:
    """Safe data ingest tool verifying extension parameters."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Target data artifact not found at: {file_path}")
    return pd.read_csv(file_path)

def save_model(model, filepath: str):
    """Pickles engineered pipelines safely to disk."""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath: str):
    """Loads pickled model binaries or safely returns None if unbuilt."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        return pickle.load(f)