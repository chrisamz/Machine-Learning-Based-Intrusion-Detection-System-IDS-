# scripts/preprocess_data.py

import pandas as pd
import pyshark

def preprocess_data(input_file='network_traffic.pcap', output_file='preprocessed_data.csv'):
    """
    Preprocess captured network traffic data.

    Args:
        input_file (str): Input file with captured traffic data.
        output_file (str): Output file to save preprocessed data.
    """
    capture = pyshark.FileCapture(input_file)
    data = []

    for packet in capture:
        try:
            packet_info = {
                'src_ip': packet.ip.src,
                'dst_ip': packet.ip.dst,
                'protocol': packet.transport_layer,
                'length': packet.length
            }
            data.append(packet_info)
        except AttributeError:
            # Skip packets that do not have IP layer
            continue

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}.")

if __name__ == "__main__":
    preprocess_data()
