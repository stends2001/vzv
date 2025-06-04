import torch
import numpy as np
from datasets import data_timepoints

def construct_graph(type: str = None, n_nodes: int = 16, number_of_graphs: int = 1, raw_data = None, threshold = 0):
    graph_dict = {}

    if type == 'identity':
        identity_loops              = torch.arange(n_nodes, device='cpu', dtype=torch.int64)
        edge_index_identity_graph   = torch.stack([identity_loops, identity_loops], dim=0) 
        edge_weight_identity_graph  = torch.tensor(np.ones(edge_index_identity_graph.shape[1]), dtype = torch.float32)
        for nn in range(number_of_graphs):
            graph_dict[nn] = (edge_index_identity_graph, edge_weight_identity_graph)

    if type == 'annual':
        print(f"Raw data shape: {raw_data.shape}")
        print(f"Data timepoints shape: {data_timepoints.shape}")
        
        # Ensure raw_data is (n_weeks, n_nodes)
        if raw_data.shape[0] != len(data_timepoints):
            if raw_data.shape[1] == len(data_timepoints):
                raw_data = raw_data.T
                print(f"Transposed raw_data to shape: {raw_data.shape}")
            else:
                raise ValueError(f"Cannot match raw_data shape {raw_data.shape} with timepoints {len(data_timepoints)}")
        
        # Add year column to timepoints for grouping
        timepoints_with_year = data_timepoints.reset_index()
        
        # Group by year and process each year
        for year, year_group in timepoints_with_year.groupby('year'):
            week_indices = year_group.index.values
            print(f"Year {year}: processing weeks {week_indices[0]} to {week_indices[-1]} ({len(week_indices)} weeks)")
            
            # Get data for this year
            yearly_data = raw_data[week_indices, :]  # Shape: (weeks_in_year, n_nodes)
            print(f"  Yearly data shape: {yearly_data.shape}")
            
            # Calculate correlation between nodes (columns)
            np_corr = np.corrcoef(yearly_data.T)  # Shape: (n_nodes, n_nodes)
            np_corr = np.nan_to_num(np_corr, nan=0.0)
            
            # Create edges
            edges = []
            weights = []
            for i in range(n_nodes):
                for j in range(n_nodes):
                    weight = np_corr[i, j] if np_corr.ndim > 1 else (1.0 if i == j else 0.0)
                    if weight >= threshold:
                        edges.append([i, j])
                        weights.append(weight)

            edge_index = torch.tensor(edges, dtype=torch.long).T  
            edge_weight = torch.tensor(weights, dtype=torch.float)
            
            print(f"  Created {len(edges)} edges")
            
            # Assign to all weeks in this year
            for week_idx in week_indices:
                graph_dict[week_idx] = (edge_index, edge_weight)

    return graph_dict
