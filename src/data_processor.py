# src/data_processor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
from typing import Optional, Tuple, Dict, List
from datetime import datetime
from .config import RANDOM_STATE, MAX_MISSING_RATIO, OUTLIER_THRESHOLD
from .utils import setup_logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Class for handling all data processing operations including loading,
    cleaning, and preprocessing of inventory data.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.data_statistics: Dict = {}
        self.required_columns = ['date', 'store', 'item', 'sales']
        setup_logging()
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from file"""
        try:
            logger.info(f"Loading data from {file_path}")
            
            # Read CSV with explicit date parsing
            df = pd.read_csv(file_path, parse_dates=['date'])
            
            # If date parsing failed, try manual conversion
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                try:
                    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
                except:
                    df['date'] = pd.to_datetime(df['date'])
            
            # Convert other columns to appropriate types
            df['store'] = df['store'].astype(int)
            df['item'] = df['item'].astype(int)
            df['sales'] = df['sales'].astype(float)
            
            # Validate data structure
            if not self.validate_data_structure(df):
                raise ValueError("Invalid data structure")
            
            # Compute statistics
            self._compute_statistics(df)
            
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def validate_data_structure(self, df: pd.DataFrame) -> bool:
        """Validate the structure of input data"""
        try:
            # Check required columns
            missing_cols = set(self.required_columns) - set(df.columns)
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False
            
            # Check for empty dataframe
            if df.empty:
                logger.error("Empty dataframe")
                return False
            
            # Check data types
            try:
                # Attempt conversion if needed
                if not pd.api.types.is_datetime64_any_dtype(df['date']):
                    df['date'] = pd.to_datetime(df['date'])
                    
                df['store'] = df['store'].astype(int)
                df['item'] = df['item'].astype(int)
                df['sales'] = df['sales'].astype(float)
            except Exception as type_error:
                logger.error(f"Data type conversion failed: {str(type_error)}")
                return False
            
            # Check for too many missing values
            missing_ratio = df.isnull().sum().max() / len(df)
            if missing_ratio > MAX_MISSING_RATIO:
                logger.error(f"Too many missing values: {missing_ratio:.2%}")
                return False
                
            return True
                
        except Exception as e:
            logger.error(f"Error in data validation: {str(e)}")
            return False
            
    def _compute_statistics(self, df: pd.DataFrame) -> None:
        """Compute and store basic statistics about the data"""
        try:
            self.data_statistics = {
                'n_rows': len(df),
                'n_stores': df['store'].nunique(),
                'n_items': df['item'].nunique(),
                'date_range': (df['date'].min(), df['date'].max()),
                'missing_values': df.isnull().sum().to_dict(),
                'numeric_statistics': df.describe().to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # in MB
                'processing_time': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error computing statistics: {str(e)}")
            raise
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the data by handling missing values and outliers"""
        try:
            logger.info("Starting data cleaning")
            df = df.copy()
            
            # Convert date column
            df['date'] = pd.to_datetime(df['date'])
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Handle outliers
            df = self._handle_outliers(df)
            
            # Validate data types
            df = self._validate_data_types(df)
            
            # Final validation
            if not self.validate_cleaned_data(df):
                raise ValueError("Data validation failed after cleaning")
            
            logger.info("Data cleaning completed")
            return df
            
        except Exception as e:
            logger.error(f"Error in data cleaning: {str(e)}")
            raise
            
    def validate_cleaned_data(self, df: pd.DataFrame) -> bool:
        """Validate data after cleaning"""
        try:
            # Check for missing values
            if df.isnull().any().any():
                logger.error("Cleaned data contains missing values")
                return False
            
            # Check for negative sales
            if (df['sales'] < 0).any():
                logger.error("Negative sales values found")
                return False
            
            # Check date continuity
            date_gaps = df.groupby(['store', 'item'])['date'].apply(
                lambda x: x.diff().dt.days.max()
            )
            if (date_gaps > 1).any():
                logger.warning("Gaps found in date sequence")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in cleaned data validation: {str(e)}")
            return False

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        try:
            df = df.copy()
            
            # Forward fill within groups for non-datetime columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = df.groupby(['store', 'item'])[col].ffill()
                df[col] = df.groupby(['store', 'item'])[col].bfill()
                
            # Fill any remaining missing values with appropriate defaults
            if 'sales' in df.columns:
                df['sales'] = df['sales'].fillna(df['sales'].mean())
                
            # Log missing value statistics
            missing_after = df.isnull().sum()
            if missing_after.any():
                logger.warning(f"Remaining missing values: {missing_after[missing_after > 0]}")
                
            return df
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR method"""
        try:
            df_clean = df.copy()
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                # Calculate bounds for each store-item combination
                bounds = df.groupby(['store', 'item'])[col].agg(
                    lambda x: self._calculate_bounds(x)
                ).to_dict()
                
                # Apply bounds
                for (store, item), group_bounds in bounds.items():
                    mask = (df['store'] == store) & (df['item'] == item)
                    df_clean.loc[mask, col] = df_clean.loc[mask, col].clip(
                        lower=group_bounds['lower'],
                        upper=group_bounds['upper']
                    )
                
            return df_clean
            
        except Exception as e:
            logger.error(f"Error handling outliers: {str(e)}")
            raise
            
    def _calculate_bounds(self, series: pd.Series) -> Dict[str, float]:
        """Calculate bounds for outlier detection"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        return {
            'lower': Q1 - OUTLIER_THRESHOLD * IQR,
            'upper': Q3 + OUTLIER_THRESHOLD * IQR
        }

    def _validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and convert data types"""
        try:
            # Ensure numeric columns are float
            numeric_cols = ['sales']
            df[numeric_cols] = df[numeric_cols].astype(float)
            
            # Ensure categorical columns are int
            categorical_cols = ['store', 'item']
            df[categorical_cols] = df[categorical_cols].astype(int)
            
            # Validate date column
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error validating data types: {str(e)}")
            raise
            
    def prepare_sequences(self, 
                        df: pd.DataFrame, 
                        sequence_length: int, 
                        target_col: str = 'sales') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for time series models"""
        try:
            sequences = []
            targets = []
            
            for (store, item), group in df.groupby(['store', 'item']):
                series = group[target_col].values
                if len(series) > sequence_length:
                    for i in range(len(series) - sequence_length):
                        sequences.append(series[i:i+sequence_length])
                        targets.append(series[i+sequence_length])
                        
            return np.array(sequences), np.array(targets)
            
        except Exception as e:
            logger.error(f"Error preparing sequences: {str(e)}")
            raise
            
    def split_data(self, 
                df: pd.DataFrame, 
                split_date: str,
                target_col: str = 'sales') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets based on date"""
        try:
            split_date = pd.to_datetime(split_date)
            train = df[df['date'] < split_date]
            test = df[df['date'] >= split_date]
            
            # Validate split
            if len(train) == 0 or len(test) == 0:
                raise ValueError(f"Invalid split date: {split_date}")
            
            logger.info(f"Train set shape: {train.shape}, Test set shape: {test.shape}")
            return train, test
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise
            
    def scale_data(self, 
                train_df: pd.DataFrame, 
                test_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Scale numerical features"""
        try:
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns
            
            # Fit scaler on training data
            train_scaled = train_df.copy()
            train_scaled[numeric_cols] = self.scaler.fit_transform(train_df[numeric_cols])
            
            if test_df is not None:
                test_scaled = test_df.copy()
                test_scaled[numeric_cols] = self.scaler.transform(test_df[numeric_cols])
                return train_scaled, test_scaled
                
            return train_scaled, None
            
        except Exception as e:
            logger.error(f"Error scaling data: {str(e)}")
            raise
            
    def get_data_info(self) -> Dict:
        """Get information about the processed data"""
        return self.data_statistics