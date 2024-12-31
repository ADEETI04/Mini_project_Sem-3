# src/utils.py
import logging
import os
from typing import Optional
import pandas as pd
import psutil
from datetime import datetime
import json
from pathlib import Path

from .config import LOG_FORMAT, LOG_FILE, LOG_LEVEL

def setup_logging(log_file: Optional[str] = None) -> None:
    """Setup logging configuration"""
    if log_file is None:
        log_file = LOG_FILE
        
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def create_directory_structure() -> None:
    """Create necessary directory structure for the project"""
    directories = [
        'data',
        'models',
        'models/checkpoints',
        'reports',
        'logs',
        'notebooks',
        'tests',
        'templates'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def validate_data_schema(df: pd.DataFrame) -> bool:
    """Validate if dataframe matches expected schema"""
    required_columns = {'date', 'store', 'item', 'sales'}
    return all(col in df.columns for col in required_columns)

def format_metrics(metrics: dict) -> dict:
    """Format metrics dictionary for display"""
    return {
        key: f"{value:.2f}" if isinstance(value, float) else str(value)
        for key, value in metrics.items()
    }

def get_memory_usage() -> str:
    """Get current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB"

def save_model_metadata(model_dir: str, metadata: dict) -> None:
    """Save model metadata including training info and performance metrics"""
    try:
        metadata_path = os.path.join(model_dir, 'model_metadata.json')
        # Add timestamp to metadata
        metadata['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
    except Exception as e:
        logging.error(f"Error saving model metadata: {str(e)}")

def load_model_metadata(model_dir: str) -> dict:
    """Load model metadata"""
    try:
        metadata_path = os.path.join(model_dir, 'model_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logging.error(f"Error loading model metadata: {str(e)}")
        return {}

def check_system_resources() -> dict:
    """Check system resources availability"""
    try:
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/')
        
        return {
            'memory_available_gb': memory.available / (1024**3),
            'memory_percent': memory.percent,
            'cpu_percent': cpu_percent,
            'disk_free_gb': disk.free / (1024**3),
            'disk_percent': disk.percent
        }
    except Exception as e:
        logging.error(f"Error checking system resources: {str(e)}")
        return {}

def cleanup_old_files(directory: str, pattern: str, max_files: int = 5) -> None:
    """Clean up old files keeping only the most recent ones"""
    try:
        files = list(Path(directory).glob(pattern))
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Remove old files
        for file in files[max_files:]:
            try:
                os.remove(file)
                logging.info(f"Removed old file: {file}")
            except Exception as e:
                logging.warning(f"Could not remove file {file}: {str(e)}")
                
    except Exception as e:
        logging.error(f"Error during cleanup: {str(e)}")

def validate_paths(*paths: str) -> bool:
    """Validate that all required paths exist"""
    try:
        for path in paths:
            if not os.path.exists(path):
                logging.error(f"Required path does not exist: {path}")
                return False
        return True
    except Exception as e:
        logging.error(f"Error validating paths: {str(e)}")
        return False

def ensure_file_permissions(filepath: str, mode: int = 0o644) -> bool:
    """Ensure file has correct permissions"""
    try:
        if os.path.exists(filepath):
            os.chmod(filepath, mode)
            return True
        return False
    except Exception as e:
        logging.error(f"Error setting file permissions: {str(e)}")
        return False

def get_project_root() -> Path:
    """Get project root directory"""
    return Path(__file__).parent.parent

def configure_environment() -> None:
    """Configure environment variables and settings"""
    try:
        # Set common environment variables
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
        
        # Create necessary directories
        create_directory_structure()
        
        # Setup logging
        setup_logging()
        
        # Check system resources
        resources = check_system_resources()
        logging.info(f"System resources: {resources}")
        
    except Exception as e:
        logging.error(f"Error configuring environment: {str(e)}")
        raise