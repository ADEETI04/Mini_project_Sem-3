# src/config.py
import os
from pathlib import Path
from datetime import datetime

# Project structure
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
REPORT_DIR = os.path.join(BASE_DIR, 'reports')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# Ensure directories exist
for directory in [DATA_DIR, MODEL_DIR, REPORT_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# Data configuration
TRAIN_FILE = 'train.csv'
DATA_PATH = os.path.join(DATA_DIR, TRAIN_FILE)

# Data validation parameters
MAX_MISSING_RATIO = 0.3
OUTLIER_THRESHOLD = 3.0

# Model parameters
RANDOM_STATE = 42
SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 7

# LSTM parameters
LSTM_PARAMS = {
    'units': [50, 100],
    'dropout': [0.1, 0.2, 0.3],
    'batch_size': [32, 64],
    'epochs': 50
}

# XGBoost parameters
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.01,
    'subsample': 0.8
}

# Feature engineering parameters
LAG_FEATURES = [1, 7, 14, 30]
ROLLING_WINDOWS = [7, 14, 30]

# Evaluation parameters
CROSS_VALIDATION_SPLITS = 5
EVALUATION_METRICS = ['MAE', 'RMSE', 'MAPE', 'R2']

# Monitoring parameters
DRIFT_THRESHOLD = 0.05
MONITORING_METRICS = ['MAE', 'RMSE', 'MAPE']
MONITORING_FREQUENCY = '1D'

# Reporting parameters
REPORT_TEMPLATE_PATH = os.path.join(BASE_DIR, 'templates', 'report_template.html')
DEFAULT_REPORT_NAME = f'forecast_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'

# Logging configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = os.path.join(LOG_DIR, 'inventory_forecast.log')
LOG_LEVEL = 'INFO'

# Validation thresholds
MAX_MISSING_RATIO = 0.3  # Maximum allowed ratio of missing values
OUTLIER_THRESHOLD = 3.0  # Number of IQRs for outlier detection
MIN_TRAINING_SAMPLES = 1000  # Minimum number of samples required for training
DATA_DRIFT_THRESHOLD = 0.05  # P-value threshold for drift detection

# Model training parameters
EARLY_STOPPING_PATIENCE = 10
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 32
MAX_EPOCHS = 100

# Database configuration (if needed)
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'inventory_forecast',
    'user': 'postgres',
    'password': 'password'
}

# API configuration
API_HOST = '0.0.0.0'
API_PORT = 8000
API_WORKERS = 4