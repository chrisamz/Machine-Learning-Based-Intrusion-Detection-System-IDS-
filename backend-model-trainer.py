# backend/model_trainer.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Constants
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def load_data(train_file: str, test_file: str):
    """
    Load training and testing data from CSV files.

    Args:
        train_file (str): The training data CSV file.
        test_file (str): The testing data CSV file.

    Returns:
        Tuple of numpy arrays: (X_train, y_train, X_test, y_test)
    """
    train_path = os.path.join(PROCESSED_DATA_DIR, train_file)
    test_path = os.path.join(PROCESSED_DATA_DIR, test_file)

    logger.info(f"Loading training data from {train_path}")
    logger.info(f"Loading testing data from {test_path}")

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    X_train = train_data.drop(columns=['length'])
    y_train = train_data['length']

    X_test = test_data.drop(columns=['length'])
    y_test = test_data['length']

    return X_train.values, y_train.values, X_test.values, y_test.values

def build_model(input_dim: int):
    """
    Build and compile a Keras Sequential model.

    Args:
        input_dim (int): The number of input features.

    Returns:
        A compiled Keras model.
    """
    model = Sequential([
        Dense(64, input_dim=input_dim, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """
    Train the Keras model.

    Args:
        X_train (np.array): Training data features.
        y_train (np.array): Training data labels.
        X_test (np.array): Testing data features.
        y_test (np.array): Testing data labels.
        epochs (int): Number of training epochs.
        batch_size (int): Size of each training batch.

    Returns:
        The trained model and the training history.
    """
    model = build_model(X_train.shape[1])

    logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2
    )

    return model, history

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test data.

    Args:
        model: The trained Keras model.
        X_test (np.array): Testing data features.
        y_test (np.array): Testing data labels.
    """
    logger.info("Evaluating model performance")

    predictions = (model.predict(X_test) > 0.5).astype("int32")

    accuracy = accuracy_score(y_test, predictions)
    logger.info(f"Accuracy: {accuracy:.4f}")

    logger.info("Classification Report:")
    logger.info(classification_report(y_test, predictions))

    logger.info("Confusion Matrix:")
    logger.info(confusion_matrix(y_test, predictions))

def save_model(model, model_name: str):
    """
    Save the trained model to disk.

    Args:
        model: The trained Keras model.
        model_name (str): The name of the model file.
    """
    model_path = os.path.join(MODEL_DIR, model_name)
    logger.info(f"Saving model to {model_path}")
    model.save(model_path)

if __name__ == "__main__":
    train_file = "network_traffic_train.csv"
    test_file = "network_traffic_test.csv"
    model_name = "intrusion_detection_model.h5"

    X_train, y_train, X_test, y_test = load_data(train_file, test_file)
    model, history = train_model(X_train, y_train, X_test, y_test, epochs=50, batch_size=32)
    evaluate_model(model, X_test, y_test)
    save_model(model, model_name)
