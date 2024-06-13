# scripts/run_alerts.py

from backend.alert_system import alert

def run_alerts(packet_info):
    """
    Run the alert system with given packet information.

    Args:
        packet_info (dict): Information about the detected packet.
    """
    alert(packet_info)

if __name__ == "__main__":
    sample_packet_info = {
        'src_ip': '192.168.1.1',
        'dst_ip': '192.168.1.100',
        'protocol': 'TCP',
        'length': 1500
    }
    run_alerts(sample_packet_info)
