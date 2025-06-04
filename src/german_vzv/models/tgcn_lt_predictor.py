import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .core import VZVGermanyModel
from datadealing.standardizer import reconstruct_timeseries
from utils.cst import token_bundeslanden


def calculate_laplacian_with_self_loop(edge_index, edge_weight, num_nodes):
    device = edge_index.device
    valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
    edge_index = edge_index[:, valid_mask]
    edge_weight = edge_weight[valid_mask]
    
    if edge_index.shape[1] == 0:
        return torch.eye(num_nodes, device=device)
    
    adj = torch.zeros(num_nodes, num_nodes, device=device)
    adj[edge_index[0], edge_index[1]] = edge_weight
    adj = adj + torch.eye(num_nodes, device=device)
    
    row_sum = adj.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = adj.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    return normalized_laplacian


class GCNLayer(nn.Module):
    def __init__(self, output_dim: int, num_nodes: int):
        super(GCNLayer, self).__init__()
        self._num_nodes = num_nodes
        self._output_dim = output_dim
        self.weights = nn.Parameter(torch.FloatTensor(1, self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain("tanh"))

    def forward(self, inputs, edge_index, edge_weight):
        laplacian = calculate_laplacian_with_self_loop(edge_index, edge_weight, self._num_nodes)
        
        if inputs.dim() == 2:
            outputs = []
            for t in range(inputs.shape[1]):
                x_t = inputs[:, t]
                ax_t = laplacian @ x_t
                out_t = torch.tanh(ax_t.unsqueeze(-1) @ self.weights)
                outputs.append(out_t.squeeze(-1))
            outputs = torch.stack(outputs, dim=1)
            return outputs
        elif inputs.dim() == 1:
            ax = laplacian @ inputs
            outputs = torch.tanh(ax.unsqueeze(-1) @ self.weights)
            return outputs.squeeze(-1)


class TGCNLongTerm(nn.Module):
    def __init__(self, node_features, hidden_dim, num_nodes, window_size, prediction_horizon=10):
        super(TGCNLongTerm, self).__init__()
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        
        self.gcn = GCNLayer(hidden_dim, num_nodes)
        self.temporal_agg = nn.Linear(window_size * hidden_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward_single_step(self, x, edge_index, edge_weight, hidden_state=None):
        gcn_out = self.gcn(x, edge_index, edge_weight)
        gcn_flat = gcn_out.reshape(self.num_nodes, -1)
        temporal_out = self.temporal_agg(gcn_flat)
        temporal_out = torch.relu(temporal_out)
        
        if hidden_state is None:
            hidden_state = torch.zeros(self.num_nodes, self.hidden_dim, device=x.device, dtype=x.dtype)
        
        hidden_state = self.gru(temporal_out, hidden_state)
        output = self.output_layer(hidden_state)
        return output.squeeze(-1), hidden_state

    def forward(self, x, edge_index, edge_weight, target_sequence=None, teacher_forcing_ratio=0.5):
        predictions = []
        hidden_state = None
        current_input = x
        
        pred, hidden_state = self.forward_single_step(current_input, edge_index, edge_weight, hidden_state)
        predictions.append(pred)
        
        for t in range(1, self.prediction_horizon):
            use_teacher_forcing = (target_sequence is not None and torch.rand(1).item() < teacher_forcing_ratio)
            
            if use_teacher_forcing and t-1 < target_sequence.shape[1]:
                next_input_val = target_sequence[:, t-1]
            else:
                next_input_val = pred.detach()
            
            new_window = torch.cat([current_input[:, 1:], next_input_val.unsqueeze(-1)], dim=1)
            current_input = new_window
            pred, hidden_state = self.forward_single_step(current_input, edge_index, edge_weight, hidden_state)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=1)
        return predictions

    def predict_sequence(self, x, edge_index, edge_weight):
        self.eval()
        with torch.no_grad():
            return self.forward(x, edge_index, edge_weight, target_sequence=None, teacher_forcing_ratio=0.0)


class TGCNLongTermModel(VZVGermanyModel):
    def __init__(self, dataloader, optimizer_class=torch.optim.Adam, criterion=torch.nn.MSELoss(),
                 node_features=1, hidden_dim=32, window_size=8, prediction_horizon=10,
                 teacher_forcing_ratio=0.5, teacher_forcing_decay=0.99, lr=0.001,
                 updates=False, device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.teacher_forcing_decay = teacher_forcing_decay
        self.initial_tf_ratio = teacher_forcing_ratio
        self.lr = lr
        self.optimizer_class = optimizer_class
        
        super().__init__(model_class=TGCNLongTerm, dataloader=dataloader, optimizer=None,
                        criterion=criterion, updates=updates, device=device)
        self.build_model()
    
    def build_model(self):
        self.model = TGCNLongTerm(node_features=self.node_features, hidden_dim=self.hidden_dim,
                                 num_nodes=self.num_nodes, window_size=self.window_size,
                                 prediction_horizon=self.prediction_horizon).to(self.device)
        self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.lr)
        
        if self.updates:
            print(f'✅ TGCN Long-term model built with {sum(p.numel() for p in self.model.parameters())} parameters')

    def train_epoch(self):
        self.model.train()
        epoch_train_loss = 0.0
        train_batches = 0

        for snapshot in self.train_loader:
            snapshot = snapshot.to(self.device)
            x = snapshot.x.to(self.device)
            edge_index = snapshot.edge_index.to(self.device)
            edge_weight = snapshot.edge_weight.to(self.device)
            y = snapshot.y.to(self.device)
            
            if y.dim() == 1:
                target_sequence = torch.zeros(self.num_nodes, self.prediction_horizon, device=y.device)
                target_sequence[:, 0] = y
            else:
                target_sequence = y
            
            self.optimizer.zero_grad()
            y_hat = self.model(x, edge_index, edge_weight, target_sequence=target_sequence,
                             teacher_forcing_ratio=self.teacher_forcing_ratio)
            loss = self.criterion(y_hat[:, 0], target_sequence[:, 0])
            loss.backward()
            self.optimizer.step()
            
            epoch_train_loss += loss.item()
            train_batches += 1

        return epoch_train_loss / train_batches

    def validate(self):
        self.model.eval()
        epoch_val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for snapshot in self.val_loader:
                x = snapshot.x.to(self.device)
                edge_index = snapshot.edge_index.to(self.device)
                edge_weight = snapshot.edge_weight.to(self.device)
                y = snapshot.y.to(self.device)
                
                if y.dim() == 1:
                    target_sequence = torch.zeros(self.num_nodes, self.prediction_horizon, device=y.device)
                    target_sequence[:, 0] = y
                else:
                    target_sequence = y
                
                y_hat = self.model(x, edge_index, edge_weight, target_sequence=None, teacher_forcing_ratio=0.0)
                loss = self.criterion(y_hat[:, 0], target_sequence[:, 0])
                epoch_val_loss += loss.item()
                val_batches += 1
                
        return epoch_val_loss / val_batches

    def train(self, epochs=100, early_stopping_patience=10, verbose=True):
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            self.teacher_forcing_ratio *= self.teacher_forcing_decay
            
            if verbose and epoch % 10 == 0:
                print(f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, TF Ratio: {self.teacher_forcing_ratio:.3f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_longterm_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch}')
                break
        
        self.model.load_state_dict(torch.load('best_longterm_model.pth'))
        if self.updates:
            print('✅ Long-term model trained and saved')
        
        self.train_history = {'train_losses': train_losses, 'val_losses': val_losses, 'best_val_loss': best_val_loss}

    def predict_long_term_with_ground_truth(self, data_loader=None, num_sequences=5):
        if data_loader is None:
            data_loader = self.test_loader
            
        self.model.eval()
        all_predictions = []
        all_inputs = []
        all_ground_truths = []
        
        count = 0
        with torch.no_grad():
            for snapshot in data_loader:
                if count >= num_sequences:
                    break
                    
                x = snapshot.x.to(self.device)
                edge_index = snapshot.edge_index.to(self.device)
                edge_weight = snapshot.edge_weight.to(self.device)
                y = snapshot.y.to(self.device)
                
                predictions = self.model.predict_sequence(x, edge_index, edge_weight)
                all_predictions.append(predictions.cpu().numpy())
                all_inputs.append(x.cpu().numpy())
                all_ground_truths.append(y.cpu().numpy())
                count += 1
        
        return np.array(all_predictions), np.array(all_inputs), np.array(all_ground_truths)

    def plot_long_term_predictions_with_reconstruction(self, county_idx=0, sequence_idx=0, added_text=""):
        predictions, inputs, ground_truths = self.predict_long_term_with_ground_truth(num_sequences=max(1, sequence_idx+1))
        
        # Reconstruct standardized values to original scale
        input_reconstructed = reconstruct_timeseries(inputs[sequence_idx], params=self.dataloader.std_params)
        pred_reconstructed = reconstruct_timeseries(predictions[sequence_idx], params=self.dataloader.std_params)
        gt_reconstructed = reconstruct_timeseries(ground_truths[sequence_idx].reshape(self.num_nodes, 1), params=self.dataloader.std_params)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Standardized values
        input_data = inputs[sequence_idx][county_idx, :]
        pred_data = predictions[sequence_idx][county_idx, :]
        gt_data = ground_truths[sequence_idx][county_idx] if ground_truths[sequence_idx].ndim > 1 else ground_truths[sequence_idx]
        
        x_input = range(len(input_data))
        x_pred = range(len(input_data), len(input_data) + len(pred_data))
        x_gt = len(input_data)  # Ground truth is at the prediction start
        
        sns.lineplot(x=x_input, y=input_data, ax=ax1, label='Input Window', color='blue', marker='o')
        sns.lineplot(x=x_pred, y=pred_data, ax=ax1, label=f'Predictions ({self.prediction_horizon} steps)', color='red', marker='s')
        ax1.scatter(x_gt, gt_data, color='green', s=100, label='Ground Truth (1-step)', zorder=5)
        
        ax1.axvline(x=len(input_data)-0.5, color='gray', linestyle='--', alpha=0.7)
        ax1.set_title(f'Standardized Values - {token_bundeslanden[county_idx]}{added_text}')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Values')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Reconstructed values
        sns.lineplot(x=x_input, y=input_reconstructed[county_idx, :], ax=ax2, label='Input Window', color='blue', marker='o')
        sns.lineplot(x=x_pred, y=pred_reconstructed[county_idx, :], ax=ax2, label=f'Predictions ({self.prediction_horizon} steps)', color='red', marker='s')
        ax2.scatter(x_gt, gt_reconstructed[county_idx], color='green', s=100, label='Ground Truth (1-step)', zorder=5)
        
        ax2.axvline(x=len(input_data)-0.5, color='gray', linestyle='--', alpha=0.7)
        ax2.set_title(f'Reconstructed Values - {token_bundeslanden[county_idx]}{added_text}')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Values')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        return fig, (ax1, ax2)