import torch
import torch.nn as nn
import torch.nn.functional as F
from .core import VZVGermanyModel
import numpy as np

def calculate_laplacian_with_self_loop(edge_index, edge_weight, num_nodes):
    """Calculate normalized Laplacian from edge_index and edge_weight"""
    device = edge_index.device
    
    # Create adjacency matrix from edge_index and edge_weight
    adj = torch.zeros(num_nodes, num_nodes, device=device)
    adj[edge_index[0], edge_index[1]] = edge_weight
    
    # Add self-loops
    adj = adj + torch.eye(num_nodes, device=device)
    
    # Calculate normalized Laplacian
    row_sum = adj.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = (
        adj.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    )
    return normalized_laplacian


class GCNLayer(nn.Module):
    def __init__(self, output_dim: int, num_nodes: int):
        super(GCNLayer, self).__init__()
        self._num_nodes = num_nodes
        self._output_dim = output_dim
        
        # Single weight for each node feature -> output_dim
        self.weights = nn.Parameter(torch.FloatTensor(1, self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain("tanh"))

    def forward(self, inputs, edge_index, edge_weight):
        # Calculate Laplacian on-the-fly
        laplacian = calculate_laplacian_with_self_loop(edge_index, edge_weight, self._num_nodes)
        
        # inputs shape: (num_nodes, window_size)
        # Apply GCN to each timestep in the window
        outputs = []
        for t in range(inputs.shape[1]):  # For each timestep in window
            x_t = inputs[:, t]  # (num_nodes,)
            ax_t = laplacian @ x_t  # (num_nodes,)
            # Transform to output dimension
            out_t = torch.tanh(ax_t.unsqueeze(-1) @ self.weights)  # (num_nodes, output_dim)
            outputs.append(out_t)
        
        # Stack outputs: (num_nodes, window_size, output_dim)
        outputs = torch.stack(outputs, dim=1)
        return outputs


class TGCN(nn.Module):
    def __init__(self, node_features, hidden_dim, num_nodes, window_size, periods=1):
        super(TGCN, self).__init__()
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.periods = periods
        
        # GCN layer
        self.gcn = GCNLayer(hidden_dim, num_nodes)
        
        # Temporal aggregation using simple linear layer
        self.temporal_agg = nn.Linear(window_size * hidden_dim, hidden_dim)
        
        # GRU layer for temporal modeling
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Output layer
        self.linear = nn.Linear(hidden_dim, 1)
        
        # Initialize hidden state
        self.hidden_state = None

    def forward(self, x, edge_index, edge_weight):
        # x shape: (num_nodes, window_size) - features for all nodes with lagged values
        
        # Apply GCN to get spatial features for each timestep
        gcn_out = self.gcn(x, edge_index, edge_weight)  # (num_nodes, window_size, hidden_dim)
        
        # Flatten temporal dimension for aggregation
        # Reshape: (num_nodes, window_size * hidden_dim)
        gcn_flat = gcn_out.reshape(self.num_nodes, -1)
        
        # Aggregate temporal information
        temporal_out = self.temporal_agg(gcn_flat)  # (num_nodes, hidden_dim)
        temporal_out = torch.relu(temporal_out)
        
        # Initialize hidden state if needed
        if self.hidden_state is None or self.hidden_state.shape[0] != self.num_nodes:
            self.hidden_state = torch.zeros(self.num_nodes, self.hidden_dim, 
                                          device=x.device, dtype=x.dtype)
        
        # Apply GRU
        self.hidden_state = self.gru(temporal_out, self.hidden_state)
        
        # Apply output layer
        output = self.linear(self.hidden_state)  # (num_nodes, 1)
        
        return output.squeeze(-1)  # (num_nodes,)

    def reset_hidden_state(self):
        """Reset hidden state - call this between different sequences"""
        self.hidden_state = None


class TGCNModel(VZVGermanyModel):
    """
    TGCN model that inherits from VZVGermanyModel core class.
    """
    def __init__(self, 
                 dataloader,
                 optimizer_class=torch.optim.Adam,
                 criterion=torch.nn.MSELoss(),
                 node_features=1,
                 hidden_dim=32,
                 window_size=8,
                 periods=1,
                 lr=0.001,
                 updates=False,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        # Store model parameters
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.periods = periods
        self.lr = lr
        self.optimizer_class = optimizer_class
        
        # Initialize parent class
        super().__init__(
            model_class=TGCN,
            dataloader=dataloader,
            optimizer=None,  # Will be set in build_model
            criterion=criterion,
            updates=updates,
            device=device
        )
        
        # Build the model
        self.build_model()
    
    def build_model(self):
        """Build the TGCN model and initialize optimizer"""
        # Create the model
        self.model = TGCN(
            node_features=self.node_features,
            hidden_dim=self.hidden_dim,
            num_nodes=self.num_nodes,  # This comes from parent class (16)
            window_size=self.window_size,
            periods=self.periods
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.lr)
        
        if self.updates:
            print(f'✅ TGCN model built with {sum(p.numel() for p in self.model.parameters())} parameters')
            print(f'   - Window size: {self.window_size}')
            print(f'   - Hidden dim: {self.hidden_dim}')
            print(f'   - Num nodes: {self.num_nodes}')

    def train_epoch(self):
        """Override train_epoch to handle hidden state reset"""
        self.model.train()
        epoch_train_loss = 0.0
        train_batches = 0

        for snapshot in self.train_loader:
            # Reset hidden state for each new sequence
            self.model.reset_hidden_state()
            
            snapshot = snapshot.to(self.device)
            x = snapshot.x.to(self.device)
            edge_index = snapshot.edge_index.to(self.device)
            edge_weight = snapshot.edge_weight.to(self.device)
            y = snapshot.y.to(self.device)
            
            # Debug print to understand data shape (only first batch)
            if train_batches == 0 and self.updates:
                print(f"Input x shape: {x.shape}")
                print(f"Target y shape: {y.shape}")
                print(f"Edge index shape: {edge_index.shape}")
                print(f"Edge weight shape: {edge_weight.shape}")
            
            # Forward pass
            self.optimizer.zero_grad()
            y_hat = self.model(x, edge_index, edge_weight)
            loss = self.criterion(y_hat, y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track loss
            epoch_train_loss += loss.item()
            train_batches += 1

        return epoch_train_loss / train_batches

    def validate(self):
        """Override validate to handle hidden state reset"""
        self.model.eval()
        epoch_val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for snapshot in self.val_loader:
                # Reset hidden state for each new sequence
                self.model.reset_hidden_state()
                
                x = snapshot.x.to(self.device)
                edge_index = snapshot.edge_index.to(self.device)
                edge_weight = snapshot.edge_weight.to(self.device)
                y = snapshot.y.to(self.device)
                
                # Forward pass
                y_hat = self.model(x, edge_index, edge_weight)
                loss = self.criterion(y_hat, y)
                
                # Track loss
                epoch_val_loss += loss.item()
                val_batches += 1
                
        return epoch_val_loss / val_batches

    def predict(self, data_loader=None):
        """Override predict to handle hidden state reset"""
        if data_loader is None:
            data_loader = self.test_loader
            
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for snapshot in data_loader:
                # Reset hidden state for each new sequence
                self.model.reset_hidden_state()
                
                x = snapshot.x.to(self.device)
                edge_index = snapshot.edge_index.to(self.device)
                edge_weight = snapshot.edge_weight.to(self.device)
                y = snapshot.y.to(self.device)
                
                # Forward pass
                y_hat = self.model(x, edge_index, edge_weight)
                
                predictions.append(y_hat.cpu().numpy())
                targets.append(y.cpu().numpy())

        return np.concatenate(predictions), np.concatenate(targets)
