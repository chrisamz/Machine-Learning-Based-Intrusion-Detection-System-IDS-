# backend/real_time_monitor.py

import os
import time
import logging
import pandas as pd
import numpy as np
import pyshark
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_FILE = "intrusion_detection_model.h5"
SCALER_FILE = "scaler.pkl"
ALERT_THRESHOLD = 0.5

# Load the trained model
model_path = os.path.join(MODEL_DIR, MODEL_FILE)
logger.info(f"Loading model from {model_path}")
model = load_model(model_path)

# Load the scaler
scaler_path = os.path.join(MODEL_DIR, SCALER_FILE)
logger.info(f"Loading scaler from {scaler_path}")
scaler = pd.read_pickle(scaler_path)

def capture_live_packets(interface="eth0", capture_duration=60):
    """
    Capture live network packets.

    Args:
        interface (str): The network interface to capture packets from.
        capture_duration (int): Duration to capture packets in seconds.

    Returns:
        pd.DataFrame: Captured packet data.
    """
    logger.info(f"Starting live packet capture on {interface} for {capture_duration} seconds")
    
    capture = pyshark.LiveCapture(interface=interface)
    capture.sniff(timeout=capture_duration)

    packet_data = []

    for packet in capture.sniff_continuously(packet_count=100):
        packet_info = {
            'length': packet.length,
            'protocol': packet.transport_layer,
            'src_ip': packet.ip.src,
            'dst_ip': packet.ip.dst,
            'src_port': packet[p].srcport,
            'dst_port': packet[p].dstport,
            'timestamp': packet.sniff_time,
        }
        packet_data.append(packet_info)
    
    return pd.DataFrame(packet_data)

def process_packets(packet_df):
    """
    Process captured packets for prediction.

    Args:
        packet_df (pd.DataFrame): DataFrame containing packet data.

    Returns:
        np.array: Processed feature array.
    """
    # Data preprocessing steps: Fill in with the appropriate preprocessing steps
    # Assuming packet_df has columns that match the features used during training

    features = packet_df.drop(columns=['timestamp', 'src_ip', 'dst_ip', 'protocol', 'src_port', 'dst_port'])
    scaled_features = scaler.transform(features)
    
    return scaled_features

def predict_intrusion(features):
    """
    Predict intrusion using the trained model.

    Args:
        features (np.array): Array of features.

    Returns:
        np.array: Prediction probabilities.
    """
    logger.info("Predicting intrusions")
    predictions = model.predict(features)
    return predictions

def monitor_network(interface="eth0", capture_duration=60):
    """
    Monitor network for intrusions in real-time.

    Args:
        interface (str): The network interface to monitor.
        capture_duration (int): Duration to capture packets in seconds.
    """
    while True:
        logger.info("Starting new monitoring cycle")
        
        packets = capture_live_packets(interface, capture_duration)
        
        if not packets.empty:
            features = process_packets(packets)
            predictions = predict_intrusion(features)
            
            for i, prediction in enumerate(predictions):
                if prediction > ALERT_THRESHOLD:
                    logger.warning(f"Intrusion detected: {packets.iloc[i].to_dict()}")
                else:
                    logger.info(f"No intrusion: {packets.iloc[i].to_dict()}")
        
        time.sleep(capture_duration)

if __name__ == "__main__":
    network_interface = "eth0"
    monitoring_duration = 60  # seconds

    monitor_network(interface=network_interface, capture_duration=monitoring_duration)
