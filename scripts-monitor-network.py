# scripts/monitor_network.py

import pyshark
import joblib
from backend.alert_system import alert

def monitor_network(interface='eth0', model_file='ids_model.pkl'):
    """
    Monitor network traffic and detect intrusions in real-time.

    Args:
        interface (str): Network interface to monitor.
        model_file (str): File with the trained model.
    """
    model = joblib.load(model_file)
    capture = pyshark.LiveCapture(interface=interface)

    print(f"Starting network monitoring on {interface}. Press Ctrl+C to stop.")
    try:
        for packet in capture.sniff_continuously():
            try:
                packet_info = {
                    'src_ip': packet.ip.src,
                    'dst_ip': packet.ip.dst,
                    'protocol': packet.transport_layer,
                    'length': packet.length
                }
                features = [[packet_info['src_ip'], packet_info['dst_ip'], packet_info['protocol'], packet_info['length']]]
                prediction = model.predict(features)
                if prediction == 1:  # Assuming '1' indicates an intrusion
                    alert(packet_info)
            except AttributeError:
                # Skip packets that do not have IP layer
                continue
    except KeyboardInterrupt:
        print("Network monitoring stopped.")

if __name__ == "__main__":
    monitor_network()
