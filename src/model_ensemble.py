# src/model_ensemble.py
from typing import Union
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, Any, Optional, List
import logging
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from .config import RANDOM_STATE
from .utils import setup_logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class ModelEnsemble:
    def __init__(self):
        setup_logging()
        self.models: Dict = {}
        self.weights: Dict = {}
        self.n_features = None

    def build_models(self):
        """Initialize all models"""
        try:
            # Create LSTM model
            self.models['lstm'] = self._build_lstm_model()
            
            # Create XGBoost model without early stopping
            self.models['xgboost'] = XGBRegressor(
                n_estimators=100,
                learning_rate=0.01,
                max_depth=6,
                objective='reg:squarederror',
                random_state=RANDOM_STATE
            )
            
            # Create LightGBM model
            self.models['lightgbm'] = LGBMRegressor(
                n_estimators=100,
                learning_rate=0.01,
                random_state=RANDOM_STATE
            )
            
            # Initialize equal weights
            total_models = len(self.models)
            self.weights = {name: 1.0/total_models for name in self.models.keys()}
            
            logger.info("All models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error building models: {str(e)}")
            raise

    def _build_lstm_model(self):
        """Build LSTM model architecture"""
        try:
            n_features = 45  # Number of features after dropping 'date' and 'sales'
            
            # Use Input layer to specify correct input shape
            model = Sequential([
                LSTM(50, activation='relu', input_shape=(1, n_features)),
                Dropout(0.2),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            return model
            
        except Exception as e:
            logger.error(f"Error building LSTM model: {str(e)}")
            raise
    
# In model_ensemble.py
    def train_model(self, model_name: str, features: Union[pd.DataFrame, np.ndarray], target: Union[pd.Series, np.ndarray]):
        """Train a specific model with periodic saving"""
        try:
            if model_name == 'lstm':
                # LSTM training
                n_samples = features.shape[0]
                n_features = features.shape[1] if len(features.shape) > 1 else 1
                X = features if isinstance(features, np.ndarray) else features.values
                y = target if isinstance(target, np.ndarray) else target.values
                X = X.reshape(n_samples, 1, n_features)
                
                # Add callback for periodic saving
                checkpoint_path = os.path.join('models', 'checkpoints', f'{model_name}_model.keras')
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                
                checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path,
                    save_best_only=True,
                    monitor='loss'
                )
                
                self.models[model_name].fit(
                    X, y,
                    epochs=50,
                    batch_size=32,
                    verbose=0,
                    callbacks=[checkpoint_callback]
                )
            else:
                # Train other models
                X = features if isinstance(features, np.ndarray) else features.values
                y = target if isinstance(target, np.ndarray) else target.values
                
                if model_name == 'xgboost':
                    self.models[model_name].fit(
                        X, y,
                        eval_set=[(X, y)],
                        verbose=False
                    )
                    # Save checkpoint
                    model_path = os.path.join('models', 'checkpoints', f'{model_name}_model.pkl')
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    pd.to_pickle(self.models[model_name], model_path)
                else:
                    self.models[model_name].fit(X, y)
                    # Save checkpoint
                    model_path = os.path.join('models', 'checkpoints', f'{model_name}_model.pkl')
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    pd.to_pickle(self.models[model_name], model_path)
                
            logger.info(f"{model_name} trained successfully")
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            raise

    def predict(self, model_name: str, features: pd.DataFrame) -> np.ndarray:
        """Generate predictions using a specific model"""
        try:
            if model_name == 'lstm':
                # Reshape for LSTM (samples, timesteps, features)
                X = features.values.reshape(features.shape[0], 1, features.shape[1])
                predictions = self.models[model_name].predict(X, verbose=0)
                return predictions.flatten()
            else:
                return self.models[model_name].predict(features)
                
        except Exception as e:
            logger.warning(f"Error predicting with {model_name}: {str(e)}")
            return None

    def ensemble_predict(self, features: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions"""
        try:
            predictions = {}
            total_weight = 0
            
            # Get predictions from each model
            for name, model in self.models.items():
                try:
                    pred = self.predict(name, features)
                    if pred is not None:
                        predictions[name] = pred
                        total_weight += self.weights.get(name, 0)
                    else:
                        logger.warning(f"Skipping {name} due to prediction failure")
                except Exception as e:
                    logger.warning(f"Error in {name} prediction: {str(e)}")
                    continue
            
            if not predictions:
                raise ValueError("No valid predictions from any model")
            
            # Normalize weights for available models
            weights = {k: v/total_weight for k, v in self.weights.items() 
                    if k in predictions}
            
            # Weighted average of available predictions
            final_pred = np.zeros_like(next(iter(predictions.values())))
            for name, pred in predictions.items():
                final_pred += pred * weights[name]
            
            return final_pred
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {str(e)}")
            raise

    def save_models(self, path: str):
        """Save all models to disk"""
        try:
            os.makedirs(path, exist_ok=True)
            
            for name, model in self.models.items():
                if name == 'lstm':
                    model.save(f"{path}/lstm_model.keras")
                else:
                    pd.to_pickle(model, f"{path}/{name}_model.pkl")
                    
            # Save weights
            pd.to_pickle(self.weights, f"{path}/ensemble_weights.pkl")
            logger.info(f"Models saved successfully to {path}")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise

    def load_models(self, path: str):
        """Load all models from disk"""
        try:
            for name in ['lstm', 'xgboost', 'lightgbm']:
                if name == 'lstm':
                    model_path = f"{path}/lstm_model.keras"
                    if os.path.exists(model_path):
                        self.models[name] = tf.keras.models.load_model(model_path)
                else:
                    model_path = f"{path}/{name}_model.pkl"
                    if os.path.exists(model_path):
                        self.models[name] = pd.read_pickle(model_path)
                    
            # Load weights
            weights_path = f"{path}/ensemble_weights.pkl"
            if os.path.exists(weights_path):
                self.weights = pd.read_pickle(weights_path)
            
            logger.info(f"Models loaded successfully from {path}")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def get_feature_importance(self, features: pd.DataFrame) -> pd.DataFrame:
        """Get feature importance from tree-based models"""
        try:
            importance_dict = {}
            
            if 'xgboost' in self.models:
                importance_dict['xgboost'] = pd.Series(
                    self.models['xgboost'].feature_importances_,
                    index=features.columns
                )
                
            if 'lightgbm' in self.models:
                importance_dict['lightgbm'] = pd.Series(
                    self.models['lightgbm'].feature_importances_,
                    index=features.columns
                )
                
            if importance_dict:
                importance_df = pd.DataFrame(importance_dict)
                importance_df['mean_importance'] = importance_df.mean(axis=1)
                return importance_df.sort_values('mean_importance', ascending=False)
                
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return pd.DataFrame()