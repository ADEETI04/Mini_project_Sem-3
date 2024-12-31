# src/evaluator.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import logging
from datetime import datetime
from .config import EVALUATION_METRICS
from .utils import setup_logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Class for model evaluation and performance analysis"""
    
    def __init__(self):
        setup_logging()
        self.metrics_history: Dict = {}
        self.best_model: Optional[str] = None
        self.evaluation_timestamp = datetime.now()
        
    def calculate_metrics(self, y_true: Union[np.ndarray, pd.Series], 
                        y_pred: Union[np.ndarray, pd.Series],
                        model_name: str = 'ensemble') -> Dict[str, float]:
        """Calculate all evaluation metrics with input validation"""
        try:
            # Convert inputs to numpy arrays
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # Validate input shapes
            if y_true.shape != y_pred.shape:
                raise ValueError(f"Shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}")
                
            # Check for NaN/Inf values
            if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
                raise ValueError("Input contains NaN values")
            if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
                raise ValueError("Input contains infinite values")
            
            # Calculate metrics with error handling
            try:
                metrics = {
                    'MAE': mean_absolute_error(y_true, y_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                    'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,
                    'R2': r2_score(y_true, y_pred)
                }
            except Exception as e:
                logger.error(f"Error calculating basic metrics: {str(e)}")
                raise
            
            # Additional metrics with error handling
            try:
                additional_metrics = self._calculate_additional_metrics(y_true, y_pred)
                metrics.update(additional_metrics)
            except Exception as e:
                logger.warning(f"Error calculating additional metrics: {str(e)}")
            
            # Store metrics history
            self.metrics_history[model_name] = {
                'metrics': metrics,
                'timestamp': self.evaluation_timestamp,
                'n_samples': len(y_true)
            }
            
            logger.info(f"Calculated metrics for {model_name}: MAE={metrics['MAE']:.2f}, "
                    f"RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.2f}%")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {model_name}: {str(e)}")
            raise
            
    def _calculate_additional_metrics(self, 
                                    y_true: np.ndarray, 
                                    y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate additional performance metrics"""
        try:
            metrics = {}
            
            # Direction accuracy
            if len(y_true) > 1:
                direction_true = np.sign(np.diff(y_true))
                direction_pred = np.sign(np.diff(y_pred))
                direction_correct = np.sum(direction_true == direction_pred)
                metrics['DirectionAccuracy'] = direction_correct / (len(y_true) - 1)
            
            # Forecast bias
            metrics['ForecastBias'] = np.mean(y_true - y_pred)
            
            # Tracking signal
            running_sum = np.cumsum(y_true - y_pred)
            running_mad = np.cumsum(np.abs(y_true - y_pred))
            with np.errstate(divide='ignore', invalid='ignore'):
                tracking_signal = running_sum / running_mad
            metrics['TrackingSignal'] = np.nanmean(tracking_signal)
            
            # Forecast accuracy
            accuracy = 100 * (1 - np.mean(np.abs((y_true - y_pred) / y_true)))
            metrics['ForecastAccuracy'] = accuracy
            
            # Over/Under forecast ratio
            over_forecasts = np.sum(y_pred > y_true)
            under_forecasts = np.sum(y_pred < y_true)
            metrics['OverUnderRatio'] = over_forecasts / under_forecasts if under_forecasts > 0 else np.inf
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating additional metrics: {str(e)}")
            raise
        
    def compare_models(self, 
                      predictions: Dict[str, np.ndarray],
                      y_true: np.ndarray) -> pd.DataFrame:
        """Compare performance of multiple models"""
        try:
            comparison = {}
            
            for model_name, y_pred in predictions.items():
                metrics = self.calculate_metrics(y_true, y_pred, model_name)
                comparison[model_name] = metrics
                
            comparison_df = pd.DataFrame(comparison).transpose()
            
            # Determine best model based on MAPE
            self.best_model = comparison_df['MAPE'].idxmin()
            logger.info(f"Best performing model: {self.best_model}")
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            raise
            
    def calculate_confidence_intervals(self,
                                    y_pred: np.ndarray,
                                    confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate confidence intervals for predictions"""
        try:
            z_score = {
                0.90: 1.645,
                0.95: 1.96,
                0.99: 2.576
            }.get(confidence_level, 1.96)
            
            std_error = np.std(y_pred)
            margin_error = z_score * std_error
            
            lower_bound = y_pred - margin_error
            upper_bound = y_pred + margin_error
            
            return lower_bound, upper_bound
            
        except Exception as e:
            logger.error(f"Error calculating confidence intervals: {str(e)}")
            raise
            
    def generate_evaluation_report(self) -> Dict:
        """Generate comprehensive evaluation report"""
        try:
            report = {
                'model_comparison': pd.DataFrame(
                    {name: data['metrics'] 
                    for name, data in self.metrics_history.items()}
                ).transpose(),
                'best_model': self.best_model,
                'summary': self._generate_summary(),
                'evaluation_time': self.evaluation_timestamp
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating evaluation report: {str(e)}")
            raise
            
    def _generate_summary(self) -> Dict:
        """Generate summary of model performance"""
        try:
            summary = {
                'best_model_metrics': self.metrics_history.get(
                    self.best_model, {}
                ).get('metrics', {}),
                'average_performance': {
                    metric: np.mean([
                        data['metrics'][metric] 
                        for data in self.metrics_history.values()
                    ])
                    for metric in EVALUATION_METRICS
                },
                'total_samples_evaluated': sum(
                    data['n_samples'] 
                    for data in self.metrics_history.values()
                )
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            raise

    def get_performance_trends(self) -> Dict[str, List[float]]:
        """Analyze performance trends over time"""
        try:
            trends = {}
            for metric in EVALUATION_METRICS:
                values = [
                    data['metrics'][metric] 
                    for data in self.metrics_history.values()
                ]
                trends[metric] = {
                    'values': values,
                    'trend': 'improving' if metric in ['R2', 'ForecastAccuracy'] and np.mean(np.diff(values)) > 0
                            else 'worsening' if np.mean(np.diff(values)) < 0
                            else 'stable'
                }
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing performance trends: {str(e)}")
            raise