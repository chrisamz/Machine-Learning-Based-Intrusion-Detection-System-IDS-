# backend/data_collector.py

import logging
import os
import pyshark

# Set up logging
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

def collect_data(interface: str, output_file: str, capture_duration: int):
    """
    Collect network traffic data using Wireshark (pyshark).

    Args:
        interface (str): The network interface to capture data from.
        output_file (str): The file to save the captured data to.
        capture_duration (int): The duration of the capture in seconds.
    """
    output_path = os.path.join(DATA_DIR, output_file)
    logger.info(f"Starting data collection on interface {interface} for {capture_duration} seconds.")
    
    capture = pyshark.LiveCapture(interface=interface, output_file=output_path)
    capture.sniff(timeout=capture_duration)
    
    logger.info(f"Data collection complete. Data saved to {output_path}.")

if __name__ == "__main__":
    interface = "eth0"  # Example interface, change as necessary
    output_file = "network_traffic.pcap"
    capture_duration = 60  # Capture for 60 seconds

    collect_data(interface, output_file, capture_duration)
