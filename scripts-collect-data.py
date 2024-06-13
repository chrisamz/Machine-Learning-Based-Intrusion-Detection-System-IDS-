# scripts/collect_data.py

import pyshark

def collect_data(interface='eth0', output_file='network_traffic.pcap'):
    """
    Capture network traffic using pyshark.

    Args:
        interface (str): Network interface to capture traffic from.
        output_file (str): Output file to save captured traffic.
    """
    capture = pyshark.LiveCapture(interface=interface, output_file=output_file)
    print(f"Starting packet capture on {interface}. Press Ctrl+C to stop.")
    try:
        capture.sniff(timeout=60)  # Capture for 60 seconds
    except KeyboardInterrupt:
        print("Packet capture stopped.")
    capture.close()
    print(f"Captured packets saved to {output_file}.")

if __name__ == "__main__":
    collect_data()
