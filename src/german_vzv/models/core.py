import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from datadealing.standardizer import reconstruct_timeseries 
from utils.cst import token_bundeslanden

class VZVGermanyModel:
    def __init__(self, 
                 model_class,
                 dataloader,
                 optimizer,
                 criterion,
                 updates = False,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):    
        
        self.device = device
        self.model_class = model_class
        self.dataloader   = dataloader
        self.train_loader = self.dataloader.train_dataset
        self.val_loader   = self.dataloader.val_dataset
        self.test_loader  = self.dataloader.test_dataset
        self.optimizer    = optimizer
        self.criterion    = criterion
        self.updates      = updates
        self.num_nodes = 16

        
        # Initialize model as None - subclasses should call build_model()
        self.model = None

    def build_model(self):
            """To be implemented by subclasses"""
            raise NotImplementedError("Subclasses must implement build_model method")

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_train_loss= 0.0
        train_batches   = 0

        for snapshot in self.train_loader:
            snapshot    = snapshot.to(self.device)
            x           = snapshot.x.to(self.device)
            edge_index  = torch.tensor(snapshot.edge_index).to(self.device)
            edge_weight = snapshot.edge_weight.to(self.device)
            y           = snapshot.y.to(self.device)
            # Forward pass
            self.optimizer.zero_grad()
            y_hat   = self.model(x, edge_index, edge_weight)

            loss    = self.criterion(y_hat, y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track loss
            epoch_train_loss+= loss.item()
            train_batches   += 1

        return epoch_train_loss /train_batches

    def validate(self):
        """Validate the model"""
        self.model.eval()
        epoch_val_loss = 0
        val_batches    = 0
        
        with torch.no_grad():
            for snapshot in self.val_loader:
                x           = snapshot.x.to(self.device)
                edge_index  = snapshot.edge_index.to(self.device)
                edge_weight = snapshot.edge_weight.to(self.device)
                y           = snapshot.y.to(self.device)                
                # Forward pass
                y_hat   = self.model(x, edge_index, edge_weight)
                loss    = self.criterion(y_hat, y)
                                # Track loss
                epoch_val_loss += loss.item()
                val_batches += 1
                
        return epoch_val_loss / val_batches
    
    def train(self, epochs=100, early_stopping_patience=10, verbose=True):
        """Full training loop with early stopping"""
        best_val_loss       = float('inf')
        patience_counter    = 0
        train_losses        = []
        val_losses          = []
        
        for epoch in range(epochs):
            train_loss  = self.train_epoch()
            val_loss    = self.validate()
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if verbose and epoch % 10 == 0:
                print(f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch}')
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        if self.updates:
            print('✅ Model trained and saved')
        self.train_history =  {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }
    
    def predict(self, data_loader=None):
        """Make predictions"""
        if data_loader is None:
            data_loader = self.test_loader
            
        self.model.eval()
        predictions = []
        targets     = []
        
        with torch.no_grad():
            for snapshot in data_loader:
                x           = snapshot.x.to(self.device)
                edge_index  = snapshot.edge_index.to(self.device)
                edge_weight = snapshot.edge_weight.to(self.device)
                y           = snapshot.y.to(self.device)    
                # Forward pass
                y_hat   = self.model(x, edge_index, edge_weight)
                loss    = self.criterion(y_hat, y)                                

                predictions.append(y_hat.cpu().numpy())
                targets.append(y.cpu().numpy())

        return np.concatenate(predictions), np.concatenate(targets)
    
    def evaluate(self, data_loader=None):
        """Evaluate model performance"""
        predictions, targets = self.predict(data_loader)
        

        # currently: (n_timesteps * num_nodes) => (num_nodes, n_timesteps)
        predictions                 = predictions.reshape(int(len(predictions)/self.num_nodes), self.num_nodes).T    
        predictions_reconstructed   = reconstruct_timeseries(predictions, params = self.dataloader.std_params)
        targets                     = targets.reshape(int(len(targets)/self.num_nodes), self.num_nodes).T
        targets_reconstructed       = reconstruct_timeseries(targets, params = self.dataloader.std_params)


        # Calculate metrics
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
   
        maes_reconstructed = []
        for county_idx in range(self.num_nodes):
            county_predictions = predictions_reconstructed[county_idx]
            county_targets     = targets_reconstructed[county_idx]
            mae = np.sqrt(mean_squared_error(county_targets, county_predictions))
            maes_reconstructed.append(mae)

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
        
    def plot_predictions(self, county_idx=4, added_text = None):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))

        labels              = self.testing_history['targets_reconstructed']
        preds               = self.testing_history['predictions_reconstructed']  

        sns.lineplot(preds[county_idx], color = 'blue', ax = ax, label = f'model')
        sns.lineplot(labels[county_idx], color = 'red', linestyle = '--', ax = ax, label = 'ground truth')

        ax.fill_between(np.arange(len(labels[county_idx])), 0, labels[county_idx], alpha = 0.1, color = 'black')
        ax.set_title(f'Predictions - {token_bundeslanden[county_idx]}{added_text}')
        ax.set_xlabel('Week')
        ax.set_ylabel('predictions')
        ax.grid(True)
        ax.legend()

        plt.tight_layout()

        return fig, ax

