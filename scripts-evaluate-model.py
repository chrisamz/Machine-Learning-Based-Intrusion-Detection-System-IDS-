# scripts/evaluate_model.py

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def evaluate_model(input_file='preprocessed_data.csv', model_file='ids_model.pkl'):
    """
    Evaluate the trained machine learning model.

    Args:
        input_file (str): Input file with preprocessed data.
        model_file (str): File with the trained model.
    """
    df = pd.read_csv(input_file)
    X = df[['src_ip', 'dst_ip', 'protocol', 'length']]
    y = df['label']

    model = joblib.load(model_file)
    y_pred = model.predict(X)

    print("Classification Report:")
    print(classification_report(y, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

if __name__ == "__main__":
    evaluate_model()
