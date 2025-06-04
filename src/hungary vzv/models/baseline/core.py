import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from datadealing.standardizer import reconstruct_timeseries 
from utils.cst import hu_tokens_countynames

class VZVHungaryBaselineModel:
    """Base class for baseline models (non-GNN approaches)"""
    
    def __init__(self, 
                 model_class,
                 dataloader,
                 model_params=None,
                 updates=False):
        
        self.model_class = model_class
        self.dataloader = dataloader
        self.model_params = model_params or {}
        self.updates = updates
        self.num_nodes = 20  # Hungary has 20 counties
        
        # Prepare data for sklearn models
        self.X_train, self.y_train = self._prepare_data(self.dataloader.train_dataset)
        self.X_val, self.y_val = self._prepare_data(self.dataloader.val_dataset)
        self.X_test, self.y_test = self._prepare_data(self.dataloader.test_dataset)
        
        # Initialize model
        self.model = None
        self.build_model()
        
    def _prepare_data(self, dataset):
        """Convert PyTorch Geometric data to sklearn format"""
        X_list = []
        y_list = []
        
        for data in dataset:
            # Flatten the node features for each time step
            # data.x shape: (num_nodes, num_features)
            X_flattened = data.x.numpy().flatten()
            y_flattened = data.y.numpy()
            
            X_list.append(X_flattened)
            y_list.append(y_flattened)
            
        return np.array(X_list), np.array(y_list)
    
    def build_model(self):
        """Build the baseline model"""
        self.model = self.model_class(**self.model_params)
        if self.updates:
            print(f'✅ {self.model.__class__.__name__} model built')
    
    def train(self, epochs=None, early_stopping_patience=None, verbose=True):
        """Train the baseline model"""
        if verbose and self.updates:
            print(f'Training {self.model.__class__.__name__}...')
            
        self.model.fit(self.X_train, self.y_train)
        
        if self.updates:
            print('✅ Model trained')
            
        # Store dummy training history for compatibility
        self.train_history = {
            'train_losses': [],
            'val_losses': [],
            'best_val_loss': 0.0
        }
    
    def predict(self, X=None):
        """Make predictions"""
        if X is None:
            X = self.X_test
            y_true = self.y_test
        else:
            y_true = None
            
        predictions = self.model.predict(X)
        
        if y_true is not None:
            return predictions, y_true
        return predictions
    
    def evaluate(self, X=None, y=None):
        """Evaluate model performance"""
        if X is None and y is None:
            predictions, targets = self.predict()
        else:
            predictions = self.predict(X)
            targets = y
        
        # Reshape predictions and targets to match GNN format
        # From (n_timesteps, num_nodes) to (num_nodes, n_timesteps)
        n_timesteps = len(predictions)
        predictions = predictions.reshape(n_timesteps, self.num_nodes).T
        targets = targets.reshape(n_timesteps, self.num_nodes).T
        
        # Reconstruct original scale
        predictions_reconstructed = reconstruct_timeseries(predictions, params=self.dataloader.std_params)
        targets_reconstructed = reconstruct_timeseries(targets, params=self.dataloader.std_params)
        
        # Calculate metrics on standardized data
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        # Calculate county-wise RMSE on reconstructed data
        maes_reconstructed = []
        for county_idx in range(self.num_nodes):
            county_predictions = predictions_reconstructed[county_idx]
            county_targets = targets_reconstructed[county_idx]
            county_rmse = np.sqrt(mean_squared_error(county_targets, county_predictions))
            maes_reconstructed.append(county_rmse)
        
        self.testing_history = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'predictions': predictions,
            'predictions_reconstructed': predictions_reconstructed,
            'targets': targets,
            'targets_reconstructed': targets_reconstructed,
            'mae_reconstructed': maes_reconstructed
        }
        
        if self.updates:
            print('✅ Model tested')
    
    def plot_predictions(self, county_idx=4, added_text=None):
        """Plot predictions vs ground truth for a specific county"""
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))
        
        labels = self.testing_history['targets_reconstructed']
        preds = self.testing_history['predictions_reconstructed']
        
        sns.lineplot(x=range(len(preds[county_idx])), y=preds[county_idx], 
                    color='blue', ax=ax, label='model')
        sns.lineplot(x=range(len(labels[county_idx])), y=labels[county_idx], 
                    color='red', linestyle='--', ax=ax, label='ground truth')
        
        ax.fill_between(np.arange(len(labels[county_idx])), 0, labels[county_idx], 
                       alpha=0.1, color='black')
        
        model_name = self.model.__class__.__name__
        county_name = hu_tokens_countynames[county_idx]
        title = f'Predictions - {county_name} ({model_name})'
        if added_text:
            title += f' {added_text}'
            
        ax.set_title(title)
        ax.set_xlabel('Week')
        ax.set_ylabel('Cases')
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        return fig, ax
