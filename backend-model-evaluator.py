# backend/model_evaluator.py

import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Constants
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_FILE = "intrusion_detection_model.h5"

def load_data(file_name: str):
    """
    Load data from a CSV file.

    Args:
        file_name (str): The data CSV file.

    Returns:
        Tuple of numpy arrays: (X, y)
    """
    data_path = os.path.join(PROCESSED_DATA_DIR, file_name)
    logger.info(f"Loading data from {data_path}")

    data = pd.read_csv(data_path)

    X = data.drop(columns=['label'])
    y = data['label']

    return X.values, y.values

def load_trained_model(model_file: str):
    """
    Load a trained Keras model from disk.

    Args:
        model_file (str): The name of the model file.

    Returns:
        The loaded Keras model.
    """
    model_path = os.path.join(MODEL_DIR, model_file)
    logger.info(f"Loading model from {model_path}")

    model = load_model(model_path)
    return model

def evaluate_model(model, X, y):
    """
    Evaluate the trained model on the data.

    Args:
        model: The trained Keras model.
        X (np.array): Data features.
        y (np.array): Data labels.
    """
    logger.info("Evaluating model performance")

    predictions = (model.predict(X) > 0.5).astype("int32")

    accuracy = accuracy_score(y, predictions)
    logger.info(f"Accuracy: {accuracy:.4f}")

    logger.info("Classification Report:")
    logger.info(classification_report(y, predictions))

    logger.info("Confusion Matrix:")
    logger.info(confusion_matrix(y, predictions))

    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y, model.predict(X))
    roc_auc = auc(fpr, tpr)

    logger.info(f"AUC: {roc_auc:.4f}")

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    test_file = "network_traffic_test.csv"

    X_test, y_test = load_data(test_file)
    model = load_trained_model(MODEL_FILE)
    evaluate_model(model, X_test, y_test)
