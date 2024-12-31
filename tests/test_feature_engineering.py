# tests/test_feature_engineering.py
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.feature_engineering import AdvancedFeatureEngineer

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        self.engineer = AdvancedFeatureEngineer()
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.sample_data = pd.DataFrame({
            'date': dates,
            'store': [1] * 100,
            'item': [1] * 100,
            'sales': np.random.normal(100, 10, 100)
        })

    def test_complete_pipeline(self):
        """Test the complete feature engineering pipeline"""
        processed_df = self.engineer.create_features(self.sample_data)
        
        # Check for basic features
        self.assertIn('year', processed_df.columns)
        self.assertIn('month', processed_df.columns)
        
        # Check for lag features
        self.assertTrue(any(col.startswith('sales_lag_') for col in processed_df.columns))
        
        # Check for rolling features
        self.assertTrue(any(col.startswith('sales_rolling_') for col in processed_df.columns))
        
        # Verify no missing values in key columns
        key_columns = ['year', 'month', 'sales']
        for col in key_columns:
            self.assertFalse(processed_df[col].isnull().any())