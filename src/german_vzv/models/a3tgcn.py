from torch_geometric_temporal.nn.recurrent import A3TGCN
import torch
import torch.nn.functional as F
from .core import VZVGermanyModel

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, periods):
        super(RecurrentGCN, self).__init__()
        self.recurrent = A3TGCN(node_features, 32, periods)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x.view(x.shape[0], 1, x.shape[1]), edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h.squeeze(-1)


class A3TGCNModel(VZVGermanyModel):
    """
    BasicGNN model that inherits from VZVGermanyModel core class.
    """
    def __init__(self, 
                 dataloader,
                 optimizer_class=torch.optim.Adam,
                 criterion=torch.nn.MSELoss(),
                 node_features=1,
                 periods = 4,
                 lr=0.001,
                 updates=False,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        # Store model parameters
        self.node_features = node_features
        self.periods = periods
        self.lr = lr
        self.optimizer_class = optimizer_class
        
        # Initialize parent class
        super().__init__(
            model_class=RecurrentGCN,
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
        self.model = RecurrentGCN(
            node_features=self.node_features,
            periods=self.periods
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.lr)
        
        if self.updates:
            print(f'✅ BasicGNN model built with {sum(p.numel() for p in self.model.parameters())} parameters')
