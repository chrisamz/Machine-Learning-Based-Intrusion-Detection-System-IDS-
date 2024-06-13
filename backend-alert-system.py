# backend/alert_system.py

import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for email alerts
SMTP_SERVER = "smtp.example.com"  # Replace with your SMTP server
SMTP_PORT = 587
EMAIL_ADDRESS = "your_email@example.com"  # Replace with your email
EMAIL_PASSWORD = "your_password"  # Replace with your email password
ALERT_RECIPIENT = "recipient_email@example.com"  # Replace with recipient email

# Log file path
ALERT_LOG_FILE = os.path.join(os.path.dirname(__file__), '..', 'logs', 'alerts.log')

# Ensure the log directory exists
os.makedirs(os.path.dirname(ALERT_LOG_FILE), exist_ok=True)

# Configure logging to file
file_handler = logging.FileHandler(ALERT_LOG_FILE)
file_handler.setLevel(logging.WARNING)
file_formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

def log_alert(packet_info):
    """
    Log the alert to a file.

    Args:
        packet_info (dict): Information about the detected packet.
    """
    logger.warning(f"Intrusion detected: {packet_info}")

def send_email_alert(packet_info):
    """
    Send an email alert.

    Args:
        packet_info (dict): Information about the detected packet.
    """
    subject = "Intrusion Detection Alert"
    body = f"Intrusion detected:\n\n{packet_info}"
    
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = ALERT_RECIPIENT
    msg['Subject'] = subject
    
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            text = msg.as_string()
            server.sendmail(EMAIL_ADDRESS, ALERT_RECIPIENT, text)
        logger.info(f"Email alert sent to {ALERT_RECIPIENT}")
    except Exception as e:
        logger.error(f"Failed to send email alert: {e}")

def alert(packet_info):
    """
    Trigger an alert by logging and sending an email.

    Args:
        packet_info (dict): Information about the detected packet.
    """
    log_alert(packet_info)
    send_email_alert(packet_info)
