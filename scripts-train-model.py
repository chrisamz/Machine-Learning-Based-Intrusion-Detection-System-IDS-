# scripts/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(input_file='preprocessed_data.csv', model_file='ids_model.pkl'):
    """
    Train a machine learning model for intrusion detection.

    Args:
        input_file (str): Input file with preprocessed data.
        model_file (str): Output file to save trained model.
    """
    df = pd.read_csv(input_file)
    X = df[['src_ip', 'dst_ip', 'protocol', 'length']]  # Features
    y = df['label']  # Target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}.")

if __name__ == "__main__":
    train_model()
