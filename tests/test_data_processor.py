# tests/test_data_processor.py
import unittest
import pandas as pd
import numpy as np
import os
from src.data_processor import DataProcessor
from datetime import datetime, timedelta

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.processor = DataProcessor()
        self.sample_data = pd.DataFrame({
            'date': [datetime.now() + timedelta(days=i) for i in range(10)],
            'store': [1] * 10,
            'item': [1] * 10,
            'sales': np.random.randint(0, 100, 10)
        })
        
    def test_load_data(self):
        """Test data loading functionality"""
        # Save sample data to temp file
        temp_file = 'temp_test.csv'
        self.sample_data.to_csv(temp_file, index=False)
        
        # Load data
        loaded_data = self.processor.load_data(temp_file)
        
        # Verify data
        self.assertEqual(len(loaded_data), len(self.sample_data))
        self.assertTrue(isinstance(loaded_data['date'].iloc[0], pd.Timestamp))
        
        # Clean up
        os.remove(temp_file)
        
    def test_handle_missing_values(self):
        """Test missing value handling"""
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[3, 'sales'] = np.nan
        cleaned_data = self.processor.clean_data(data_with_missing)  # Use clean_data instead
        self.assertEqual(cleaned_data.isnull().sum().sum(), 0)