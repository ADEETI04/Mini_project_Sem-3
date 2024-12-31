# src/visualizer.py
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from .utils import setup_logging

logger = logging.getLogger(__name__)

class Visualizer:
    """Class for creating visualizations of forecasting results and analysis"""
    
    def __init__(self):
        setup_logging()
        self.plots_config = {
            'template': 'plotly_white',
            'height': 600
        }
        
    def plot_time_series(self,
                        df: pd.DataFrame,
                        date_col: str = 'date',
                        value_col: str = 'sales',
                        title: str = 'Time Series Plot') -> go.Figure:
        """Create interactive time series plot"""
        try:
            fig = px.line(df, 
                         x=date_col, 
                         y=value_col,
                         title=title,
                         template=self.plots_config['template'])
            
            fig.update_layout(
                height=self.plots_config['height'],
                showlegend=True,
                xaxis_title='Date',
                yaxis_title='Sales'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating time series plot: {str(e)}")
            raise
            
    def plot_forecast_comparison(self,
                               actual: np.ndarray,
                               predicted: Dict[str, np.ndarray],
                               dates: pd.DatetimeIndex,
                               confidence_intervals: Optional[Dict] = None) -> go.Figure:
        """Plot comparison of actual vs predicted values with confidence intervals"""
        try:
            fig = go.Figure()
            
            # Plot actual values
            fig.add_trace(go.Scatter(
                x=dates,
                y=actual,
                name='Actual',
                line=dict(color='black', width=2)
            ))
            
            # Plot predictions for each model
            colors = px.colors.qualitative.Set1
            for i, (model_name, pred) in enumerate(predicted.items()):
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=pred,
                    name=f'{model_name} Forecast',
                    line=dict(color=colors[i % len(colors)])
                ))
                
                # Add confidence intervals if provided
                if confidence_intervals and model_name in confidence_intervals:
                    lower, upper = confidence_intervals[model_name]
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=upper,
                        fill=None,
                        mode='lines',
                        line_color='rgba(0,0,0,0)',
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=lower,
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(0,0,0,0)',
                        name=f'{model_name} CI'
                    ))
                    
            fig.update_layout(
                title='Forecast Comparison',
                xaxis_title='Date',
                yaxis_title='Sales',
                height=self.plots_config['height'],
                template=self.plots_config['template']
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating forecast comparison plot: {str(e)}")
            raise
            
    def plot_feature_importance(self,
                              importance_df: pd.DataFrame,
                              top_n: int = 10) -> go.Figure:
        """Plot feature importance"""
        try:
            top_features = importance_df.nlargest(top_n, 'importance')
            
            fig = px.bar(top_features,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title=f'Top {top_n} Most Important Features')
            
            fig.update_layout(
                height=self.plots_config['height'],
                template=self.plots_config['template'],
                yaxis={'categoryorder': 'total ascending'}
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating feature importance plot: {str(e)}")
            raise
            
    def plot_model_performance(self,
                             metrics_df: pd.DataFrame) -> go.Figure:
        """Create subplot of model performance metrics"""
        try:
            metrics = metrics_df.columns.tolist()
            n_metrics = len(metrics)
            
            fig = make_subplots(
                rows=n_metrics,
                cols=1,
                subplot_titles=metrics
            )
            
            for i, metric in enumerate(metrics, 1):
                fig.add_trace(
                    go.Bar(
                        x=metrics_df.index,
                        y=metrics_df[metric],
                        name=metric
                    ),
                    row=i,
                    col=1
                )
                
            fig.update_layout(
                height=300 * n_metrics,
                template=self.plots_config['template'],
                showlegend=False,
                title_text="Model Performance Comparison"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating model performance plot: {str(e)}")
            raise
            
    def plot_residuals_analysis(self,
                              y_true: np.ndarray,
                              y_pred: np.ndarray) -> go.Figure:
        """Create residuals analysis plots"""
        try:
            residuals = y_true - y_pred
            
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=[
                    'Residuals Distribution',
                    'Residuals vs Predicted',
                    'Q-Q Plot',
                    'Residuals Over Time'
                ]
            )
            
            # Residuals distribution
            fig.add_trace(
                go.Histogram(x=residuals, name='Residuals'),
                row=1,
                col=1
            )
            
            # Residuals vs Predicted
            fig.add_trace(
                go.Scatter(
                    x=y_pred,
                    y=residuals,
                    mode='markers',
                    name='Residuals vs Predicted'
                ),
                row=1,
                col=2
            )
            
            # Q-Q plot
            from scipy import stats
            qq = stats.probplot(residuals)
            fig.add_trace(
                go.Scatter(
                    x=qq[0][0],
                    y=qq[0][1],
                    mode='markers',
                    name='Q-Q Plot'
                ),
                row=2,
                col=1
            )
            
            # Residuals over time
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(residuals)),
                    y=residuals,
                    mode='lines',
                    name='Residuals Over Time'
                ),
                row=2,
                col=2
            )
            
            fig.update_layout(
                height=800,
                template=self.plots_config['template'],
                title_text="Residuals Analysis"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating residuals analysis plot: {str(e)}")
            raise