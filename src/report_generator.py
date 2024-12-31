# src/report_generator.py
import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import jinja2
from typing import Dict, Any, Optional, List  # Added List import
import plotly.graph_objects as go
from .config import REPORT_TEMPLATE_PATH, DEFAULT_REPORT_NAME
from .utils import setup_logging

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Class for generating comprehensive forecast reports"""
    
    def __init__(self):
        setup_logging()
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader('templates')
        )
        self.report_data: Dict = {}
        
    def generate_report(self,
                    forecast_results: Dict[str, Any],
                    model_performance: Dict[str, Any],
                    monitoring_data: Optional[Dict] = None,
                    output_path: Optional[str] = None) -> str:
        """Generate complete forecast report"""
        try:
            self.report_data = {
                'timestamp': datetime.now(),
                'forecast_results': forecast_results,
                'model_performance': model_performance,
                'monitoring_data': monitoring_data
            }
            
            # Validate required data
            if 'forecast' not in forecast_results:
                raise ValueError("Forecast results must contain 'forecast' key")
            
            # Generate report sections
            self._add_executive_summary()
            self._add_forecast_analysis()
            self._add_model_performance_analysis()
            if monitoring_data:
                self._add_monitoring_analysis()
                
            # Generate HTML report
            report_html = self._render_report()
            
            # Save report
            if output_path is None:
                output_path = os.path.join('reports', DEFAULT_REPORT_NAME)
                
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_html)
                
            logger.info(f"Report generated successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise

    def _add_executive_summary(self) -> None:
        """Add executive summary section to report"""
        forecast_results = self.report_data['forecast_results']
        model_performance = self.report_data['model_performance']
        
        summary = {
            'forecast_horizon': len(forecast_results.get('forecast', [])),
            'best_model': model_performance.get('metrics', {}).get('best_model', 'N/A'),
            'accuracy_metrics': {
                metric: value for metric, value in 
                model_performance.get('metrics', {}).items()
                if metric in ['MAPE', 'R2']
            },
            'key_findings': self._generate_key_findings()
        }
        
        self.report_data['executive_summary'] = summary
        
    def _add_forecast_analysis(self) -> None:
        """Add detailed forecast analysis section"""
        forecast_results = self.report_data['forecast_results']
        
        analysis = {
            'forecast_stats': {
                'mean': forecast_results['forecast'].mean(),
                'std': forecast_results['forecast'].std(),
                'min': forecast_results['forecast'].min(),
                'max': forecast_results['forecast'].max()
            },
            'trend_analysis': self._analyze_trend(forecast_results['forecast']),
            'seasonality': self._analyze_seasonality(forecast_results['forecast'])
        }
        
        self.report_data['forecast_analysis'] = analysis
        
    def _add_model_performance_analysis(self) -> None:
        """Add detailed model performance analysis"""
        model_performance = self.report_data['model_performance']
        
        analysis = {
            'metrics_comparison': pd.DataFrame(
                model_performance.get('metrics_history', {'ensemble': model_performance['metrics']})
            ),
            'best_model': model_performance.get('best_model', 'ensemble'),
            'residuals_analysis': self._analyze_residuals(
                self.report_data['forecast_results'].get('actual', []),
                self.report_data['forecast_results'].get('forecast', [])
            )
        }
        
        self.report_data['model_analysis'] = analysis
        
    def _add_monitoring_analysis(self) -> None:
        """Add model monitoring analysis section"""
        monitoring_data = self.report_data['monitoring_data']
        
        analysis = {
            'drift_summary': self._summarize_drift(monitoring_data.get('drift', {})),
            'performance_trends': self._analyze_performance_trends(
                monitoring_data.get('performance_history', {})
            ),
            'alerts_summary': self._summarize_alerts(
                monitoring_data.get('alerts', {})
            )
        }
        
        self.report_data['monitoring_analysis'] = analysis
        
    def _summarize_drift(self, drift_results: Dict) -> Dict:
        """Summarize data drift analysis"""
        try:
            if not drift_results:
                return {}
                
            summary = {
                'total_features_checked': len(drift_results),
                'features_with_drift': sum(
                    1 for feature_drift in drift_results.values()
                    if feature_drift.get('drift_detected', False)
                ),
                'drift_details': {}
            }
            
            # Analyze drift by feature type
            for feature, drift_info in drift_results.items():
                if drift_info.get('drift_detected', False):
                    summary['drift_details'][feature] = {
                        'p_value': drift_info.get('p_value', 0),
                        'statistic': drift_info.get('ks_statistic', 0)
                    }
            
            # Calculate drift severity
            summary['drift_severity'] = (
                summary['features_with_drift'] / summary['total_features_checked']
                if summary['total_features_checked'] > 0 else 0
            )
            
            # Add interpretation
            summary['interpretation'] = self._interpret_drift_severity(
                summary['drift_severity']
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing drift: {str(e)}")
            return {}
        
    def _interpret_drift_severity(self, severity: float) -> str:
        """Interpret drift severity level"""
        if severity <= 0.1:
            return "Low data drift - Model performance stable"
        elif severity <= 0.3:
            return "Moderate data drift - Monitor model performance closely"
        else:
            return "High data drift - Consider model retraining"
    
    def _generate_key_findings(self) -> List[str]:
        """Generate key findings from the analysis"""
        findings = []
        model_performance = self.report_data['model_performance']
        
        # Add accuracy finding
        best_mape = model_performance.get('best_model_metrics', {}).get('MAPE', 0)
        findings.append(
            f"Model achieves {best_mape:.2f}% mean absolute percentage error"
        )
        
        # Add trend finding
        forecast_results = self.report_data['forecast_results']
        trend = np.polyfit(
            range(len(forecast_results['forecast'])),
            forecast_results['forecast'],
            1
        )[0]
        trend_direction = "upward" if trend > 0 else "downward"
        findings.append(
            f"Forecast shows {trend_direction} trend of {abs(trend):.2f} units per period"
        )
        
        return findings
        
    def _render_report(self) -> str:
        """Render the final HTML report"""
        template = self.template_env.get_template('report_template.html')
        return template.render(
            report_data=self.report_data,
            generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
    @staticmethod
    def _analyze_trend(data: np.ndarray) -> Dict:
        """Analyze trend in the data"""
        x = np.arange(len(data))
        trend = np.polyfit(x, data, 1)
        return {
            'slope': trend[0],
            'intercept': trend[1],
            'direction': 'increasing' if trend[0] > 0 else 'decreasing'
        }
        
    @staticmethod
    def _analyze_seasonality(data: np.ndarray) -> Dict:
        """Analyze seasonality in the data"""
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Ensure data is valid for decomposition
        if len(data) < 2:
            return {
                'seasonal_strength': 0,
                'period': None,
                'peaks': []
            }
            
        try:
            decomposition = seasonal_decompose(
                data,
                period=min(7, len(data)-1),  # Weekly seasonality or shorter
                extrapolate_trend='freq'
            )
            
            return {
                'seasonal_strength': float(np.std(decomposition.seasonal)),
                'period': 7,
                'peaks': list(pd.Series(decomposition.seasonal).nlargest(3).index)
            }
        except Exception as e:
            logger.warning(f"Error in seasonality analysis: {str(e)}")
            return {
                'seasonal_strength': 0,
                'period': None,
                'peaks': []
            }
            
    def _rank_models(self, metrics_history: Dict) -> pd.DataFrame:
        """Rank models based on performance metrics"""
        rankings = pd.DataFrame(metrics_history)
        rankings['overall_score'] = rankings.mean(axis=1)
        return rankings.sort_values('overall_score', ascending=False)
        
    def _analyze_residuals(self, actual: np.ndarray, predicted: np.ndarray) -> Dict:
        """Analyze prediction residuals"""
        if len(actual) == 0 or len(predicted) == 0:
            return {}
            
        residuals = actual - predicted
        return {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'skew': float(stats.skew(residuals)),
            'kurtosis': float(stats.kurtosis(residuals)),
            'normality_test': self._test_normality(residuals)  # This is correct now
        }
        
    @staticmethod  # Change this from instance method to static method
    def _test_normality(data: np.ndarray) -> Dict:
        """Test residuals for normality"""
        try:
            statistic, p_value = stats.normaltest(data)
            return {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'is_normal': p_value > 0.05
            }
        except Exception:
            return {
                'statistic': 0.0,
                'p_value': 0.0,
                'is_normal': False
            }
            
    # After _test_normality method, add these methods:

    def _analyze_performance_trends(self, performance_history: Dict) -> Dict:
        """Analyze trends in performance metrics"""
        try:
            if not performance_history:
                return {}
                
            trends = {}
            for metric, values in performance_history.items():
                if isinstance(values, list) and values:
                    metric_values = [v.get('value', 0) if isinstance(v, dict) else v for v in values]
                    if metric_values:
                        trends[metric] = {
                            'current': metric_values[-1],
                            'mean': float(np.mean(metric_values)),
                            'std': float(np.std(metric_values)),
                            'trend': 'improving' if len(metric_values) > 1 and
                                    metric_values[-1] < metric_values[0] else 'worsening',
                            'change_percent': float(((metric_values[-1] - metric_values[0]) / 
                                                metric_values[0] * 100)
                                                if len(metric_values) > 1 and metric_values[0] != 0 
                                                else 0)
                        }
            
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing performance trends: {str(e)}")
            return {}

    def _summarize_alerts(self, alerts: Dict) -> Dict:
        """Summarize monitoring alerts"""
        try:
            if not alerts:
                return {}
                
            summary = {
                'total_alerts': sum(len(alert_list) for alert_list in alerts.values()),
                'alerts_by_type': {
                    alert_type: len(alert_list)
                    for alert_type, alert_list in alerts.items()
                },
                'recent_alerts': []
            }
            
            # Get most recent alerts
            all_alerts = []
            for alert_type, alert_list in alerts.items():
                for alert in alert_list:
                    if isinstance(alert, dict):
                        all_alerts.append({
                            **alert,
                            'type': alert_type,
                            'timestamp': alert.get('timestamp', datetime.now())
                        })
            
            # Sort by timestamp and get recent ones
            sorted_alerts = sorted(
                all_alerts,
                key=lambda x: x['timestamp'],
                reverse=True
            )
            summary['recent_alerts'] = sorted_alerts[:5]
            
            # Add severity levels
            summary['severity_distribution'] = {
                'high': len([a for a in all_alerts if 'drift' in a.get('type', '')]),
                'medium': len([a for a in all_alerts if 'degradation' in a.get('type', '')]),
                'low': len([a for a in all_alerts if not any(x in a.get('type', '') 
                                                            for x in ['drift', 'degradation'])])
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing alerts: {str(e)}")
            return {}

    def generate_visualization(self, 
                            actual: np.ndarray, 
                            predicted: np.ndarray, 
                            dates: np.ndarray) -> Dict[str, Any]:
        """Generate visualization for the report"""
        try:
            fig = go.Figure()
            
            # Add actual values
            fig.add_trace(go.Scatter(
                x=dates,
                y=actual,
                name='Actual',
                line=dict(color='blue', width=2)
            ))
            
            # Add predictions
            fig.add_trace(go.Scatter(
                x=dates,
                y=predicted,
                name='Predicted',
                line=dict(color='red', width=2)
            ))
            
            # Calculate confidence intervals
            std_dev = np.std(predicted - actual)
            upper_bound = predicted + 1.96 * std_dev
            lower_bound = predicted - 1.96 * std_dev
            
            # Add confidence intervals
            fig.add_trace(go.Scatter(
                x=dates,
                y=upper_bound,
                fill=None,
                mode='lines',
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=lower_bound,
                fill='tonexty',
                mode='lines',
                line=dict(color='rgba(0,0,0,0)'),
                name='95% CI'
            ))
            
            # Update layout
            fig.update_layout(
                title='Forecast vs Actual Values',
                xaxis_title='Date',
                yaxis_title='Value',
                hovermode='x unified',
                showlegend=True
            )
            
            return {
                'forecast_plot': fig,
                'confidence_intervals': {
                    'upper': upper_bound.tolist(),
                    'lower': lower_bound.tolist()
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
            return {}