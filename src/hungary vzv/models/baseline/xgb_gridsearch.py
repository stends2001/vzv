import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from .core import VZVHungaryBaselineModel
import time
from typing import Dict, List, Any
import json

class XGBoostGridSearch:
    def __init__(self, 
                 dataloader,
                 param_grid: Dict[str, List[Any]] = None,
                 scoring: str = 'rmse',
                 cv_folds: int = 3,
                 verbose: bool = True,
                 n_jobs: int = -1,
                 random_state: int = 42):
        
        self.dataloader = dataloader
        self.scoring = scoring
        self.cv_folds = cv_folds
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.num_nodes = 20
        
        if param_grid is None:
            self.param_grid = self._get_default_param_grid()
        else:
            self.param_grid = param_grid
            
        # Prepare data
        self.X_train, self.y_train = self._prepare_data(self.dataloader.train_dataset)
        self.X_val, self.y_val = self._prepare_data(self.dataloader.val_dataset)
        self.X_test, self.y_test = self._prepare_data(self.dataloader.test_dataset)
        
        # Results storage
        self.results_ = []
        self.best_params_ = None
        self.best_score_ = float('inf')
        self.best_model_ = None
    
    def _get_default_param_grid(self):
        return {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [0, 0.1],
            'reg_lambda': [0, 0.1]
        }
    
    def _prepare_data(self, dataset):
        X_list = []
        y_list = []
        
        for data in dataset:
            X_flattened = data.x.numpy().flatten()
            y_flattened = data.y.numpy()
            X_list.append(X_flattened)
            y_list.append(y_flattened)
            
        return np.array(X_list), np.array(y_list)
    
# Add these methods to the XGBoostGridSearch class

    def _create_cv_splits(self):
        """Create time-series cross-validation splits"""
        n_samples = len(self.X_train)
        fold_size = n_samples // self.cv_folds
        
        cv_splits = []
        for i in range(self.cv_folds):
            train_end = (i + 1) * fold_size
            if i == self.cv_folds - 1:
                train_end = n_samples
                
            val_start = max(0, train_end - fold_size)
            
            train_idx = list(range(0, val_start))
            val_idx = list(range(val_start, train_end))
            
            if len(train_idx) > 0 and len(val_idx) > 0:
                cv_splits.append((train_idx, val_idx))
                
        return cv_splits

    def _evaluate_params(self, params):
        """Evaluate single parameter combination"""
        start_time = time.time()
        
        model_params = {
            **params,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'objective': 'reg:squarederror'
        }
        
        # Cross-validation
        cv_splits = self._create_cv_splits()
        cv_scores = []
        
        for train_idx, val_idx in cv_splits:
            X_train_fold = self.X_train[train_idx]
            y_train_fold = self.y_train[train_idx]
            X_val_fold = self.X_train[val_idx]
            y_val_fold = self.y_train[val_idx]
            
            model = xgb.XGBRegressor(**model_params)
            model.fit(X_train_fold, y_train_fold)
            
            y_pred = model.predict(X_val_fold)
            
            if self.scoring == 'rmse':
                score = np.sqrt(mean_squared_error(y_val_fold, y_pred))
            else:
                score = mean_squared_error(y_val_fold, y_pred)
                
            cv_scores.append(score)
        
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)
        
        # Final model on validation set
        final_model = xgb.XGBRegressor(**model_params)
        final_model.fit(self.X_train, self.y_train)
        
        val_pred = final_model.predict(self.X_val)
        if self.scoring == 'rmse':
            val_score = np.sqrt(mean_squared_error(self.y_val, val_pred))
        else:
            val_score = mean_squared_error(self.y_val, val_pred)
        
        elapsed_time = time.time() - start_time
        
        return {
            'params': params,
            'mean_cv_score': mean_cv_score,
            'std_cv_score': std_cv_score,
            'val_score': val_score,
            'cv_scores': cv_scores,
            'model': final_model,
            'training_time': elapsed_time
        }

    # Add this method to XGBoostGridSearch class

    def fit(self):
        """Perform grid search"""
        param_combinations = list(ParameterGrid(self.param_grid))
        total_combinations = len(param_combinations)
        
        if self.verbose:
            print(f"Starting grid search with {total_combinations} combinations...")
            print(f"Using {self.cv_folds}-fold cross-validation")
            print("-" * 60)
        
        for i, params in enumerate(param_combinations):
            if self.verbose:
                print(f"[{i+1}/{total_combinations}] Testing: {params}")
            
            try:
                result = self._evaluate_params(params)
                self.results_.append(result)
                
                if result['val_score'] < self.best_score_:
                    self.best_score_ = result['val_score']
                    self.best_params_ = params.copy()
                    self.best_model_ = result['model']
                
                if self.verbose:
                    print(f"  CV Score: {result['mean_cv_score']:.4f}")
                    print(f"  Val Score: {result['val_score']:.4f}")
                    print(f"  Best So Far: {self.best_score_:.4f}")
                    print("-" * 40)
                    
            except Exception as e:
                if self.verbose:
                    print(f"  Error: {str(e)}")
                continue
        
        if self.verbose:
            print(f"\nBest parameters: {self.best_params_}")
            print(f"Best score: {self.best_score_:.4f}")


    # Add these methods to XGBoostGridSearch class

    def get_results_dataframe(self):
        """Get results as DataFrame"""
        if not self.results_:
            return pd.DataFrame()
        
        rows = []
        for result in self.results_:
            row = {
                'mean_cv_score': result['mean_cv_score'],
                'std_cv_score': result['std_cv_score'],
                'val_score': result['val_score'],
                'training_time': result['training_time']
            }
            row.update(result['params'])
            rows.append(row)
        
        return pd.DataFrame(rows).sort_values('val_score')

    def save_results(self, filepath):
        """Save results to JSON file"""
        results_to_save = []
        for result in self.results_:
            result_copy = result.copy()
            result_copy.pop('model', None)  # Remove model for serialization
            results_to_save.append(result_copy)
        
        save_data = {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'results': results_to_save
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)

    def get_best_model(self):
        """Get optimized model in baseline framework"""
        if self.best_model_ is None:
            raise ValueError("No best model found. Run fit() first.")
        
        from .xgb import XGBoostModel
        
        # Create model with best parameters
        best_model = XGBoostModel(
            dataloader=self.dataloader,
            **self.best_params_,
            updates=self.verbose
        )
        
        # Replace with fitted model
        best_model.model = self.best_model_
        
        return best_model
