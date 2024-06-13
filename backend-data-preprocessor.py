# backend/data_preprocessing.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pyshark
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Constants
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

def parse_pcap_to_csv(pcap_file: str, output_csv: str):
    """
    Parse pcap file and convert it to a CSV file.

    Args:
        pcap_file (str): The pcap file to be parsed.
        output_csv (str): The output CSV file name.
    """
    pcap_path = os.path.join(RAW_DATA_DIR, pcap_file)
    output_path = os.path.join(PROCESSED_DATA_DIR, output_csv)
    
    logger.info(f"Parsing pcap file {pcap_path} and converting to CSV.")

    capture = pyshark.FileCapture(pcap_path)
    packets = []

    for packet in capture:
        packet_info = {
            'timestamp': packet.sniff_time,
            'source': packet.ip.src if hasattr(packet, 'ip') else np.nan,
            'destination': packet.ip.dst if hasattr(packet, 'ip') else np.nan,
            'protocol': packet.transport_layer if hasattr(packet, 'transport_layer') else np.nan,
            'length': packet.length
        }
        packets.append(packet_info)
    
    df = pd.DataFrame(packets)
    df.to_csv(output_path, index=False)

    logger.info(f"CSV file saved to {output_path}")

def preprocess_data(input_csv: str, output_train: str, output_test: str, test_size: float = 0.2):
    """
    Preprocess the data for machine learning.

    Args:
        input_csv (str): The input CSV file containing raw data.
        output_train (str): The output file name for the training data.
        output_test (str): The output file name for the testing data.
        test_size (float): The proportion of the dataset to include in the test split.
    """
    input_path = os.path.join(PROCESSED_DATA_DIR, input_csv)
    output_train_path = os.path.join(PROCESSED_DATA_DIR, output_train)
    output_test_path = os.path.join(PROCESSED_DATA_DIR, output_test)
    
    logger.info(f"Preprocessing data from {input_path}")

    df = pd.read_csv(input_path)
    
    # Handle missing values
    df.fillna(0, inplace=True)

    # Encode categorical variables
    df = pd.get_dummies(df, columns=['protocol'])

    # Feature scaling
    scaler = StandardScaler()
    df[['length']] = scaler.fit_transform(df[['length']])

    # Define features and labels
    X = df.drop(columns=['timestamp', 'source', 'destination'])
    y = (df['length'] > df['length'].mean()).astype(int)  # Example label (can be customized)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Save the processed data
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    train_data.to_csv(output_train_path, index=False)
    test_data.to_csv(output_test_path, index=False)

    logger.info(f"Training data saved to {output_train_path}")
    logger.info(f"Testing data saved to {output_test_path}")

if __name__ == "__main__":
    pcap_file = "network_traffic.pcap"
    raw_csv = "network_traffic_raw.csv"
    train_csv = "network_traffic_train.csv"
    test_csv = "network_traffic_test.csv"

    parse_pcap_to_csv(pcap_file, raw_csv)
    preprocess_data(raw_csv, train_csv, test_csv)
