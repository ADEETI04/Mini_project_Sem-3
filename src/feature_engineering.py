import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pandas.tseries.holiday import USFederalHolidayCalendar
import logging
from src.config import LAG_FEATURES, ROLLING_WINDOWS

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """Advanced feature engineering for time series data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.holiday_calendar = USFederalHolidayCalendar()
        
    # def create_features(self, df):
    #     """Complete feature engineering pipeline"""
    #     try:
    #         df = df.copy()
            
    #         # Basic time features
    #         df = self._add_time_features(df)
            
    #         # Cyclical features
    #         df = self._add_cyclical_features(df)
            
    #         # Lag features
    #         df = self._add_lag_features(df)
            
    #         # Statistical features
    #         df = self._add_statistical_features(df)
            
    #         # Holiday features
    #         df = self._add_holiday_features(df)
            
    #         # Scale features
    #         df = self._scale_features(df)
            
    #         logger.info(f"Created {len(df.columns)} features successfully")
    #         return df
            
    #     except Exception as e:
    #         logger.error(f"Error in feature engineering: {str(e)}")
    #         raise


    def create_features(self, df):
        """Complete feature engineering pipeline"""
        try:
            # Create a deep copy to avoid modifying original data
            df = df.copy()
            
            # Store sales column separately
            sales = df['sales'].copy() if 'sales' in df.columns else None
            
            # Handle infinite values before scaling
            df = self._handle_infinite_values(df)
            
            # Basic time features
            df = self._add_time_features(df)
            
            # Cyclical features
            df = self._add_cyclical_features(df)
            
            # Lag features
            df = self._add_lag_features(df)
            
            # Statistical features
            df = self._add_statistical_features(df)
            
            # Holiday features
            df = self._add_holiday_features(df)
            
            # Handle any infinities created during feature engineering
            df = self._handle_infinite_values(df)
            
            # Scale features
            df = self._scale_features(df)
            
            # Restore sales column if it existed
            if sales is not None:
                df['sales'] = sales
            
            logger.info(f"Created {len(df.columns)} features successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise

    def _handle_infinite_values(self, df):
        """Handle infinite values in numerical columns"""
        try:
            # Get numerical columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            # Replace infinite values with NaN
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
            
            # Handle NaN values
            for col in numeric_cols:
                # Fill NaN with median for numeric columns
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                
                # Clip extreme values
                q1 = df[col].quantile(0.01)
                q3 = df[col].quantile(0.99)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                df[col] = df[col].clip(lower_bound, upper_bound)
                
            return df
        except Exception as e:
            logger.error(f"Error handling infinite values: {str(e)}")
            raise

    def _scale_features(self, df):
        """Scale numerical features"""
        try:
            # Get numerical columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            # Columns to exclude from scaling
            exclude_cols = ['sales', 'store', 'item', 'year', 'month', 'day_of_month', 
                        'day_of_week', 'week_of_year', 'is_weekend', 'is_holiday']
            
            # Get columns to scale
            scale_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            if scale_cols:
                # Store original values
                original_values = df[scale_cols].copy()
                
                # Scale features
                df[scale_cols] = self.scaler.fit_transform(original_values)
                
                # Check for any NaN or infinite values after scaling
                if df[scale_cols].isna().any().any() or np.isinf(df[scale_cols]).any().any():
                    logger.warning("Scaling produced NaN or infinite values. Reverting to original values.")
                    df[scale_cols] = original_values
            
            return df
            
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            raise
    
    def _validate_features(self, df):
        """Validate the engineered features"""
        try:
            # Check for required columns
            required_cols = ['sales', 'store', 'item', 'date']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Check for infinite values
            if np.isinf(df.select_dtypes(include=[np.number])).any().any():
                raise ValueError("Infinite values found in features")
            
            # Check for NaN values
            if df.isna().any().any():
                raise ValueError("NaN values found in features")
                
            return True
            
        except Exception as e:
            logger.error(f"Feature validation failed: {str(e)}")
            return False
                
    def _add_time_features(self, df):
        """Extract time-based features"""
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        return df
        
    def _add_cyclical_features(self, df):
        """Add cyclical encoding of time features"""
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
        return df
        
    def _add_lag_features(self, df):
        """Create lag features"""
        for lag in LAG_FEATURES:
            df[f'sales_lag_{lag}'] = df.groupby(['store', 'item'])['sales'].shift(lag)
            
        return df
        
    def _add_statistical_features(self, df):
        """Create statistical features"""
        for window in ROLLING_WINDOWS:
            # Basic rolling statistics
            df[f'sales_rolling_mean_{window}'] = df.groupby(['store', 'item'])['sales'].transform(
                lambda x: x.rolling(window, min_periods=1).mean())
            df[f'sales_rolling_std_{window}'] = df.groupby(['store', 'item'])['sales'].transform(
                lambda x: x.rolling(window, min_periods=1).std())
            df[f'sales_rolling_max_{window}'] = df.groupby(['store', 'item'])['sales'].transform(
                lambda x: x.rolling(window, min_periods=1).max())
            df[f'sales_rolling_min_{window}'] = df.groupby(['store', 'item'])['sales'].transform(
                lambda x: x.rolling(window, min_periods=1).min())
            
            # Advanced rolling statistics
            df[f'sales_rolling_skew_{window}'] = df.groupby(['store', 'item'])['sales'].transform(
                lambda x: x.rolling(window, min_periods=1).skew())
            df[f'sales_rolling_kurt_{window}'] = df.groupby(['store', 'item'])['sales'].transform(
                lambda x: x.rolling(window, min_periods=1).kurt())
                
            # Expanding statistics
            df[f'sales_expanding_mean_{window}'] = df.groupby(['store', 'item'])['sales'].transform(
                lambda x: x.expanding(min_periods=1).mean())
                
            # Percentage changes
            df[f'sales_pct_change_{window}'] = df.groupby(['store', 'item'])['sales'].transform(
                lambda x: x.pct_change(periods=window))
        
        return df
        
    def _add_holiday_features(self, df):
        """Add holiday-related features"""
        # Get holidays
        holidays = self.holiday_calendar.holidays(
            start=df['date'].min(),
            end=df['date'].max()
        )
        
        # Create holiday flags
        df['is_holiday'] = df['date'].isin(holidays).astype(int)
        
        # Days until next and since last holiday
        holiday_dates = pd.Series(holidays)
        df['days_to_holiday'] = df['date'].apply(
            lambda x: min((holiday_dates - x)[holiday_dates > x].dt.days.min(), 7)
            if len((holiday_dates - x)[holiday_dates > x]) > 0 else 7)
        df['days_from_holiday'] = df['date'].apply(
            lambda x: min((x - holiday_dates)[holiday_dates < x].dt.days.min(), 7)
            if len((x - holiday_dates)[holiday_dates < x]) > 0 else 7)
            
        return df
        
    def _scale_features(self, df):
        """Scale numerical features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df
        
    def get_feature_importance(self, model, feature_names):
        """Calculate feature importance using model"""
        try:
            if hasattr(model, 'feature_importances_'):
                self.feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
            return self.feature_importance
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return None