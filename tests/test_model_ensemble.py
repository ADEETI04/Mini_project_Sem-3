# tests/test_model_ensemble.py
import unittest
import numpy as np
import pandas as pd
from src.model_ensemble import ModelEnsemble

class TestModelEnsemble(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.ensemble = ModelEnsemble()
        np.random.seed(42)
        
        # Create sample data
        self.X_train = np.random.randn(100, 5)
        self.y_train = np.random.randn(100)
        self.X_test = np.random.randn(20, 5)
        self.y_test = np.random.randn(20)

        # Convert to DataFrame/Series
        self.X_train = pd.DataFrame(self.X_train, columns=[f'feature_{i}' for i in range(5)])
        self.y_train = pd.Series(self.y_train)

    def test_build_models(self):
        """Test model initialization"""
        self.ensemble.build_models()
        expected_models = ['xgboost', 'lightgbm']
        for model_name in expected_models:
            self.assertIn(model_name, self.ensemble.models)

    def test_train_model(self):
        """Test model training"""
        self.ensemble.build_models()
        model_name = 'xgboost'
        self.ensemble.train_model(model_name, self.X_train, self.y_train)
        self.assertTrue(hasattr(self.ensemble.models[model_name], 'predict'))

    def test_ensemble_predict(self):
        """Test ensemble predictions"""
        self.ensemble.build_models()
        
        # Train models
        for model_name in ['xgboost', 'lightgbm']:
            self.ensemble.train_model(model_name, self.X_train, self.y_train)
        
        # Create test DataFrame with index
        X_test_df = pd.DataFrame(
            self.X_test, 
            columns=[f'feature_{i}' for i in range(5)],
            index=pd.date_range('2023-01-01', periods=len(self.X_test))
        )
        
        # Get predictions
        predictions = self.ensemble.ensemble_predict(X_test_df)
        
        # Verify output
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(isinstance(predictions, np.ndarray))