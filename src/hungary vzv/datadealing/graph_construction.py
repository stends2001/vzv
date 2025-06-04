import torch
import numpy as np
from datasets import data_timepoints

def construct_graph(type: str = None, n_nodes: int = 20, number_of_graphs: int = 1, raw_data = None, threshold = 0):
    graph_dict = {}

    if type == 'identity':
        identity_loops              = torch.arange(n_nodes, device='cpu', dtype=torch.int64)
        edge_index_identity_graph   = torch.stack([identity_loops, identity_loops], dim=0) 
        edge_weight_identity_graph  = torch.tensor(np.ones(edge_index_identity_graph.shape[1]), dtype = torch.float32)
        for nn in range(number_of_graphs):
            graph_dict[nn] = (edge_index_identity_graph, edge_weight_identity_graph)


    return graph_dict
