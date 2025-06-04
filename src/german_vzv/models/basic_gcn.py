import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from .core import VZVGermanyModel


class BasicGNN(torch.nn.Module):
    """
    A basic GNN model with two convolutional layers.
    """
    def __init__(self, input_dim, hidden_dim=32, dropout=0.25, name='BasicGNN'):
        super(BasicGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)
        self.dropout = dropout
        self.name = name
        
    def forward(self, x, edge_index, edge_weight=None):
        # First convolutional layer
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second convolutional layer
        x = self.conv2(x, edge_index, edge_weight)
        
        return x.view(-1)


class BasicGNNModel(VZVGermanyModel):
    """
    BasicGNN model that inherits from VZVGermanyModel core class.
    """
    def __init__(self, 
                 dataloader,
                 optimizer_class=torch.optim.Adam,
                 criterion=torch.nn.MSELoss(),
                 input_dim=1,
                 hidden_dim=32,
                 dropout=0.25,
                 lr=0.001,
                 updates=False,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        # Store model parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.optimizer_class = optimizer_class
        
        # Initialize parent class
        super().__init__(
            model_class=BasicGNN,
            dataloader=dataloader,
            optimizer=None,  # Will be set in build_model
            criterion=criterion,
            updates=updates,
            device=device
        )
        
        # Build the model
        self.build_model()
    
    def build_model(self):
        """Build the BasicGNN model and initialize optimizer"""
        # Create the model
        self.model = BasicGNN(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.lr)
        
        if self.updates:
            print(f'✅ BasicGNN model built with {sum(p.numel() for p in self.model.parameters())} parameters')
