import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import json
from scipy import stats
from .config import DRIFT_THRESHOLD, MONITORING_METRICS
from .utils import setup_logging

logger = logging.getLogger(__name__)

class ModelMonitor:
    """Class for monitoring model performance and data drift"""
    
    def __init__(self):
        setup_logging()
        self.performance_history: Dict = {}
        self.drift_history: Dict = {}
        self.alerts: Dict = {}
        
    def track_model_performance(self,
                            metrics: Dict[str, float],
                            timestamp: Optional[datetime] = None) -> None:
        """Track model performance metrics over time"""
        try:
            if timestamp is None:
                timestamp = datetime.now()
                
            for metric, value in metrics.items():
                if metric not in self.performance_history:
                    self.performance_history[metric] = []
                    
                self.performance_history[metric].append({
                    'timestamp': timestamp,
                    'value': value
                })
                
            self._check_performance_degradation()
            logger.info("Performance metrics tracked successfully")
            
        except Exception as e:
            logger.error(f"Error tracking performance: {str(e)}")
            raise
            
    def detect_data_drift(self,
                        reference_data: pd.DataFrame,
                        current_data: pd.DataFrame,
                        columns: Optional[List[str]] = None) -> Dict:
        """Detect data drift between reference and current data"""
        try:
            drift_results = {}
            
            if columns is None:
                columns = reference_data.select_dtypes(
                    include=[np.number]).columns
                    
            for column in columns:
                # Perform Kolmogorov-Smirnov test
                ks_statistic, p_value = stats.ks_2samp(
                    reference_data[column].dropna(),
                    current_data[column].dropna()
                )
                
                drift_detected = p_value < DRIFT_THRESHOLD
                
                drift_results[column] = {
                    'ks_statistic': ks_statistic,
                    'p_value': p_value,
                    'drift_detected': drift_detected
                }
                
                if drift_detected:
                    self._create_alert(
                        f"Data drift detected in column {column}",
                        'data_drift'
                    )
                    
            self.drift_history[datetime.now()] = drift_results
            return drift_results
            
        except Exception as e:
            logger.error(f"Error detecting data drift: {str(e)}")
            raise
            
    def _check_performance_degradation(self) -> None:
        """Check for model performance degradation"""
        for metric in MONITORING_METRICS:
            if metric in self.performance_history:
                values = [entry['value'] 
                        for entry in self.performance_history[metric]]
                
                if len(values) >= 3:
                    # Check for consistent degradation
                    if all(values[-i] > values[-i-1] 
                        for i in range(1, min(4, len(values)))):
                        self._create_alert(
                            f"Performance degradation detected in {metric}",
                            'performance_degradation'
                        )
                        
    def _create_alert(self,
                    message: str,
                    alert_type: str) -> None:
        """Create and store monitoring alert"""
        timestamp = datetime.now()
        if alert_type not in self.alerts:
            self.alerts[alert_type] = []
            
        self.alerts[alert_type].append({
            'timestamp': timestamp,
            'message': message
        })
        
        logger.warning(f"Alert created: {message}")
        
    def get_monitoring_report(self) -> Dict:
        """Generate comprehensive monitoring report"""
        report = {
            'performance_history': self.performance_history,
            'drift_history': self.drift_history,
            'alerts': self.alerts,
            'summary': self._generate_monitoring_summary()
        }
        return report
        
    def _generate_monitoring_summary(self) -> Dict:
        """Generate summary of monitoring metrics"""
        summary = {
            'total_alerts': sum(len(alerts) for alerts in self.alerts.values()),
            'performance_trends': {},
            'drift_frequency': {}
        }
        
        # Calculate performance trends
        for metric in MONITORING_METRICS:
            if metric in self.performance_history:
                values = [entry['value'] 
                         for entry in self.performance_history[metric]]
                if values:
                    summary['performance_trends'][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'trend': 'increasing' if len(values) > 1 and
                                values[-1] > values[0] else 'decreasing'
                    }
                
        # Calculate drift frequency
        if self.drift_history:
            n_checks = len(self.drift_history)
            n_drifts = sum(
                any(col['drift_detected'] 
                    for col in check.values())
                for check in self.drift_history.values()
            )
            summary['drift_frequency'] = n_drifts / n_checks if n_checks > 0 else 0
            
        return summary
        
    def save_monitoring_state(self, filepath: str) -> None:
        """Save monitoring state to file"""
        try:
            state = {
                'performance_history': {
                    metric: [{**entry, 'timestamp': entry['timestamp'].isoformat()}
                            for entry in entries]
                    for metric, entries in self.performance_history.items()
                },
                'drift_history': {
                    k.isoformat(): v 
                    for k, v in self.drift_history.items()
                },
                'alerts': {
                    alert_type: [{**alert, 'timestamp': alert['timestamp'].isoformat()}
                                for alert in alerts]
                    for alert_type, alerts in self.alerts.items()
                }
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=4)
                
            logger.info(f"Monitoring state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving monitoring state: {str(e)}")
            raise
            
    def load_monitoring_state(self, filepath: str) -> None:
        """Load monitoring state from file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
                
            self.performance_history = {
                metric: [{**entry, 'timestamp': datetime.fromisoformat(entry['timestamp'])}
                        for entry in entries]
                for metric, entries in state['performance_history'].items()
            }
            
            self.drift_history = {
                datetime.fromisoformat(k): v 
                for k, v in state['drift_history'].items()
            }
            
            self.alerts = {
                alert_type: [{**alert, 'timestamp': datetime.fromisoformat(alert['timestamp'])}
                            for alert in alerts]
                for alert_type, alerts in state['alerts'].items()
            }
            
            logger.info(f"Monitoring state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading monitoring state: {str(e)}")
            raise