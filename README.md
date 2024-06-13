# Machine Learning-Based Intrusion Detection System (IDS)

## Objective

The objective of this project is to develop an Intrusion Detection System (IDS) that leverages machine learning techniques to detect and classify network intrusions. The system will collect and preprocess network traffic data, train and evaluate various machine learning models, and provide real-time monitoring and alerts for network intrusions.

## Features

1. **Data Collection and Preprocessing**:
   - Collect network traffic data using Wireshark.
   - Preprocess the collected data to prepare it for machine learning model training.
   - Feature extraction and selection from network traffic data.

2. **Machine Learning Model Training and Evaluation**:
   - Train various machine learning models (e.g., Decision Trees, Random Forests, SVM, Neural Networks) for intrusion detection.
   - Evaluate model performance using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

3. **Real-time Monitoring and Alert System**:
   - Monitor network traffic in real-time.
   - Classify network traffic using the trained models.
   - Generate alerts for detected intrusions.

## Technologies

- **Programming Language**: Python
- **Machine Learning Libraries**: scikit-learn, TensorFlow, Keras
- **Data Collection Tool**: Wireshark
- **Data Preprocessing**: pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

## Project Structure

```
machine-learning-ids/
│
├── backend/
│   ├── __init__.py
│   ├── data_collector.py
│   ├── data_preprocessor.py
│   ├── model_trainer.py
│   ├── model_evaluator.py
│   ├── real_time_monitor.py
│   └── alert_system.py
│
├── scripts/
│   ├── collect_data.py
│   ├── preprocess_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── monitor_network.py
│   └── run_alerts.py
│
├── tests/
│   ├── __init__.py
│   ├── test_data_collector.py
│   ├── test_data_preprocessor.py
│   ├── test_model_trainer.py
│   ├── test_model_evaluator.py
│   ├── test_real_time_monitor.py
│   └── test_alert_system.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── data_exploration.ipynb
│   └── model_training.ipynb
│
├── requirements.txt
├── README.md
└── setup.py
```

## Setup Instructions

1. **Clone the Repository**:
   ```
   git clone https://github.com/yourusername/machine-learning-ids.git
   cd machine-learning-ids
   ```

2. **Create and Activate a Virtual Environment**:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the Required Packages**:
   ```
   pip install -r requirements.txt
   ```

4. **Set Up Wireshark for Data Collection**:
   - Install Wireshark from [here](https://www.wireshark.org/).
   - Configure Wireshark to capture network traffic data and save it to a file.

## Usage

### Data Collection and Preprocessing

1. **Collect Network Traffic Data**:
   ```
   python scripts/collect_data.py --output data/raw/network_traffic.pcap
   ```

2. **Preprocess Collected Data**:
   ```
   python scripts/preprocess_data.py --input data/raw/network_traffic.pcap --output data/processed/processed_data.csv
   ```

### Model Training and Evaluation

3. **Train Machine Learning Models**:
   ```
   python scripts/train_model.py --input data/processed/processed_data.csv --model_output models/
   ```

4. **Evaluate Trained Models**:
   ```
   python scripts/evaluate_model.py --model_path models/ --input data/processed/processed_data.csv
   ```

### Real-time Monitoring and Alert System

5. **Monitor Network in Real-Time**:
   ```
   python scripts/monitor_network.py --model_path models/best_model.pkl
   ```

6. **Run Alert System**:
   ```
   python scripts/run_alerts.py
   ```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any feature additions, bug fixes, or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

Special thanks to the authors and contributors of the machine learning libraries and tools used in this project, as well as the cybersecurity community for their research and resources on intrusion detection systems.
