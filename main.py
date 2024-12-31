# main.py
import argparse
import logging
import os
import json  # Add this import
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

from src.data_processor import DataProcessor
from src.feature_engineering import AdvancedFeatureEngineer
from src.model_ensemble import ModelEnsemble
from src.evaluator import ModelEvaluator
from src.monitoring import ModelMonitor
from src.report_generator import ReportGenerator
from src.visualizer import Visualizer
from src.utils import setup_logging, create_directory_structure
from src.config import (
    DATA_PATH, MODEL_DIR, REPORT_DIR, LOG_DIR,
    CROSS_VALIDATION_SPLITS, FORECAST_HORIZON
)

# Setup logging
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Advanced Inventory Forecasting System'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default=DATA_PATH,
        help='Path to training data'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default=MODEL_DIR,
        help='Directory to save models'
    )
    parser.add_argument(
        '--report_dir',
        type=str,
        default=REPORT_DIR,
        help='Directory to save reports'
    )
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Whether to perform hyperparameter optimization'
    )
    parser.add_argument(
        '--cross_validate',
        action='store_true',
        help='Whether to perform cross-validation'
    )
    return parser.parse_args()

def prepare_data_for_models(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare data for model training"""
    try:
        # Split features and target
        features = df.drop(['date', 'sales'], axis=1)
        target = df['sales']
        
        return features, target
        
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise

def train_and_optimize_models(train_data: pd.DataFrame, optimize: bool = False) -> Dict:
    """Train and optionally optimize models"""
    try:
        # Initialize components
        ensemble = ModelEnsemble()
        
        # Build models
        ensemble.build_models()
        
        # Prepare data - ensure we're not passing the 'date' column
        features = train_data.drop(['date', 'sales'], axis=1) if 'date' in train_data.columns else train_data.drop(['sales'], axis=1)
        target = train_data['sales']
        
        # Train models
        logger.info("Training models...")
        for model_name in ensemble.models:
            # All models use the same features now
            ensemble.train_model(model_name, features, target)
            
        return {
            'ensemble': ensemble,
            'features': features,
            'target': target
        }
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise

def main():
    """Main execution function"""
    try:
        # Parse arguments and setup environment
        args = parse_arguments()
        create_directory_structure()
        setup_logging()
        
        logger.info("Starting inventory forecasting system")
        
        # Initialize components
        processor = DataProcessor()
        engineer = AdvancedFeatureEngineer()
        evaluator = ModelEvaluator()
        monitor = ModelMonitor()
        reporter = ReportGenerator()
        visualizer = Visualizer()
        
        # Load and process data
        logger.info("Loading data...")
        df = processor.load_data(args.data_path)
        
        if not processor.validate_data_structure(df):
            raise ValueError("Invalid data structure")
        
        logger.info("Cleaning data...")
        df = processor.clean_data(df)
        
        logger.info("Engineering features...")
        df = engineer.create_features(df)
        
        if 'sales' not in df.columns:
            raise ValueError("Sales column missing after feature engineering")
        
        # Split data for training
        train_data = df[df['date'] < '2017-12-01']
        test_data = df[df['date'] >= '2017-12-01']
        
        if len(train_data) == 0 or len(test_data) == 0:
            raise ValueError("Invalid data split - check date ranges")
            
        logger.info(f"Train set shape: {train_data.shape}, Test set shape: {test_data.shape}")
        
        # Train models
        logger.info("Training models...")
        training_results = train_and_optimize_models(train_data, args.optimize)
        
        if training_results is None:
            raise ValueError("Model training failed")
            
        # Generate predictions
        features_test = test_data.drop(['date', 'sales'], axis=1)
        target_test = test_data['sales']
        
        predictions = training_results['ensemble'].ensemble_predict(features_test)
        
        if predictions is None:
            raise ValueError("Model prediction failed")
        
        # Evaluate performance
        metrics = evaluator.calculate_metrics(
            y_true=target_test,
            y_pred=predictions,
            model_name='ensemble'
        )
        logger.info(f"Model performance metrics: {metrics}")
        
        # Monitor for drift
        monitor.track_model_performance(metrics)
        drift_results = monitor.detect_data_drift(train_data, test_data)
        
        # Generate visualization
        visualization_results = visualizer.plot_forecast_comparison(
            actual=target_test,
            predicted={'Ensemble': predictions},
            dates=test_data['date']
        )
        
        # Generate report
        report_path = reporter.generate_report(
            forecast_results={
                'forecast': predictions,
                'actual': target_test,
                'dates': test_data['date']
            },
            model_performance={
                'metrics': metrics,
                'metrics_history': evaluator.metrics_history,
                'best_model': evaluator.best_model
            },
            monitoring_data={
                'drift': drift_results,
                'alerts': monitor.alerts,
                'performance_history': monitor.performance_history
            }
        )
        
        logger.info(f"Report saved successfully at: {report_path}")
        
        # Save models
        logger.info("Saving models...")
        training_results['ensemble'].save_models(args.model_dir)
        
        # Feature importance analysis
        if hasattr(training_results['ensemble'].models['xgboost'], 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': features_test.columns,
                'importance': training_results['ensemble'].models['xgboost'].feature_importances_
            })
            importance_df = importance_df.sort_values('importance', ascending=False)
            logger.info(f"Top 5 important features:\n{importance_df.head()}")
        
        # Performance summary
        performance_summary = {
            'accuracy': metrics['MAPE'],
            'r2_score': metrics['R2'],
            'forecast_bias': metrics.get('ForecastBias', 0),
            'data_drift_detected': any(drift_results.values()),
            'model_version': '1.0.0',
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save performance summary
        summary_path = os.path.join(args.report_dir, 'performance_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(performance_summary, f, indent=4)
        
        logger.info("Successfully completed training and evaluation")
        
        return {
            'metrics': metrics,
            'report_path': report_path,
            'model_dir': args.model_dir,
            'summary_path': summary_path
        }
        
    except Exception as e:
        logger.error(f"Application Error: {str(e)}")
        logger.error("Please check the logs for more details.")
        raise

if __name__ == "__main__":
    main()