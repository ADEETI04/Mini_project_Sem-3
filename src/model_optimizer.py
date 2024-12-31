import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
import optuna
import logging
from src.config import CROSS_VALIDATION_SPLITS

logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Advanced model optimization using grid search and Optuna"""
    
    def __init__(self):
        self.best_params = {}
        self.optimization_history = {}
        
    def optimize_model(self, model, param_grid, X_train, y_train, method='grid'):
        """Optimize model hyperparameters using specified method"""
        try:
            if method == 'grid':
                return self._grid_search(model, param_grid, X_train, y_train)
            elif method == 'optuna':
                return self._optuna_optimize(model, param_grid, X_train, y_train)
            else:
                raise ValueError(f"Unknown optimization method: {method}")
                
        except Exception as e:
            logger.error(f"Error in model optimization: {str(e)}")
            raise
            
    def _grid_search(self, model, param_grid, X_train, y_train):
        """Perform grid search with time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=CROSS_VALIDATION_SPLITS)
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_absolute_percentage_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params['grid_search'] = grid_search.best_params_
        self.optimization_history['grid_search'] = {
            'cv_results': grid_search.cv_results_,
            'best_score': grid_search.best_score_
        }
        
        return grid_search.best_estimator_
        
    def _optuna_optimize(self, model, param_grid, X_train, y_train):
        """Perform hyperparameter optimization using Optuna"""
        def objective(trial):
            params = {}
            for param, values in param_grid.items():
                if isinstance(values, list):
                    if isinstance(values[0], int):
                        params[param] = trial.suggest_int(param, min(values), max(values))
                    elif isinstance(values[0], float):
                        params[param] = trial.suggest_float(param, min(values), max(values))
                        
            model.set_params(**params)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=CROSS_VALIDATION_SPLITS)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_fold_train = X_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_train = y_train.iloc[train_idx]
                y_fold_val = y_train.iloc[val_idx]
                
                model.fit(X_fold_train, y_fold_train)
                pred = model.predict(X_fold_val)
                score = np.mean(np.abs((y_fold_val - pred) / y_fold_val))
                scores.append(score)
                
            return np.mean(scores)
            
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)
        
        self.best_params['optuna'] = study.best_params
        self.optimization_history['optuna'] = {
            'study': study,
            'best_score': study.best_value
        }
        
        # Return model with best parameters
        model.set_params(**study.best_params)
        model.fit(X_train, y_train)
        return model
        
    def plot_optimization_history(self, method='grid'):
        """Plot optimization history"""
        import matplotlib.pyplot as plt
        
        if method == 'grid':
            results = self.optimization_history['grid_search']['cv_results_']
            plt.figure(figsize=(10, 6))
            plt.plot(results['mean_test_score'])
            plt.title('Grid Search Optimization History')
            plt.xlabel('Iteration')
            plt.ylabel('Mean Test Score')
            plt.show()
            
        elif method == 'optuna':
            study = self.optimization_history['optuna']['study']
            optuna.visualization.plot_optimization_history(study)
            optuna.visualization.plot_parallel_coordinate(study)