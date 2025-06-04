from .standardizer import *
from .graph_construction import *
from torch_geometric.data import Data
import torch
from utils import *

class DataLoader:

    def __init__(self, 
                 raw_data: np.ndarray,
                 val_ratio: float = 0.2, 
                 test_ratio: float = 0.2,
                 standardization_method: str = 'zscore',
                 window_size: int = 1,
                 include_weeks: bool = False,
                 graph_type: str = 'identity',
                 threshold = 0
                 ):
        self.raw_data                   = raw_data
        self.n_nodes, self.n_timepoints = self.raw_data.shape
        self.include_weeks              = include_weeks
        self.graph_type = graph_type


        number_of_graphs = self.n_timepoints - window_size
        graph_dict       = construct_graph(graph_type, self.n_nodes, number_of_graphs, self.raw_data, threshold)

        test_size       = int(self.n_timepoints * test_ratio)
        trainval_size   = self.n_timepoints - test_size
        val_size        = int(trainval_size * val_ratio)
        train_size      = trainval_size - val_size

        initial_split = self.split(train_size, val_size, test_size)
        std_split     = self.standardize_split_on_train(initial_split, standardization_method)  
        split_windowed_data = self.construct_windows(std_split, window_size, train_size, val_size)
        self.initialize_loaders(split_windowed_data, graph_dict)


    def standardize_split_on_train(self, initial_split, standardization_method):
        train_raw       = initial_split['train']
        val_raw         = initial_split['val']
        test_raw        = initial_split['test']

        _, std_params   = standardize_casenumbers(train_raw, method=standardization_method)
        train_std, _    = standardize_casenumbers(train_raw, method=standardization_method, params=std_params)
        val_std, _      = standardize_casenumbers(val_raw, method=standardization_method, params=std_params)
        test_std, _     = standardize_casenumbers(test_raw, method=standardization_method, params=std_params)

        merged_std = np.concatenate([train_std, val_std, test_std], axis=1)

        std_split = {
            'train': train_std,
            'val': val_std,
            'test': test_std,
            'std_params': std_params,
            'merged_std': merged_std
        }

        self.std_params = std_params
        return std_split

    def split(self,train_size, val_size, test_size):
        train_raw       = self.raw_data[:, :train_size]
        val_raw         = self.raw_data[:, train_size:train_size+val_size]
        test_raw        = self.raw_data[:, train_size+val_size:]   

        ini_split = {
            'train':    train_raw,
            'val':      val_raw,
            'test':     test_raw,
        }
        return ini_split 

    def construct_windows(self, std_split, window_size, train_size, val_size):
        X_all = []
        y_all = []
        actual_time_indices = []
        split_indices = [] 

        for t in range(window_size, self.n_timepoints):
            
            case_features       = std_split['merged_std'][:, t-window_size:t]

            if self.include_weeks:
                print('weeks not currently supported')
                pass

            #     week_features       =  week_ns_st[:,t].reshape(-1,1)
            #     features  = np.hstack((case_features,week_features))

            else:
                features  = case_features

            target      = std_split['merged_std'][:, t]
            
            # Determine which split this window belongs to
            if t < train_size:
                split = 'train'
            elif t < train_size + val_size:
                split = 'val'
            else:
                split = 'test'
            
            X_all.append(features)
            y_all.append(target)
            split_indices.append(split)
            actual_time_indices.append(t)

        X_all = np.array(X_all)
        y_all = np.array(y_all)

        train_indices   = [i for i in range(len(split_indices)) if split_indices[i] == 'train']
        val_indices     = [i for i in range(len(split_indices)) if split_indices[i] == 'val']
        test_indices    = [i for i in range(len(split_indices)) if split_indices[i] == 'test']

        X_train = np.array([X_all[i] for i in range(len(split_indices)) if split_indices[i] == 'train'],  dtype=np.float32)
        y_train = np.array([y_all[i] for i in range(len(split_indices)) if split_indices[i] == 'train'],  dtype=np.float32)

        X_val = np.array([X_all[i] for i in range(len(split_indices)) if split_indices[i] == 'val'],  dtype=np.float32)
        y_val = np.array([y_all[i] for i in range(len(split_indices)) if split_indices[i] == 'val'],  dtype=np.float32)

        X_test = np.array([X_all[i] for i in range(len(split_indices)) if split_indices[i] == 'test'],  dtype=np.float32)
        y_test = np.array([y_all[i] for i in range(len(split_indices)) if split_indices[i] == 'test'],  dtype=np.float32)    

        split_windowed_data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices,
        }

        return split_windowed_data

    def initialize_loaders(self, split_windowed_data, graph_dict):
        X_train = split_windowed_data['X_train']
        y_train = split_windowed_data['y_train']
        X_val   = split_windowed_data['X_val']
        y_val   = split_windowed_data['y_val']
        X_test  = split_windowed_data['X_test']
        y_test  = split_windowed_data['y_test']
        train_indices = split_windowed_data['train_indices']
        val_indices   = split_windowed_data['val_indices']
        test_indices  = split_windowed_data['test_indices']

        train_data_list = []
        for i in range(len(X_train)):
            t           = train_indices[i]  # Get actual 

            data = Data(
                x=torch.FloatTensor(X_train[i]),
                y=torch.FloatTensor(y_train[i]),
                edge_index = graph_dict[t][0],
                edge_weight= graph_dict[t][1]                

            )
            train_data_list.append(data)

        # For validation data
        val_data_list = []
        for i in range(len(X_val)):
            t           = val_indices[i]  # Get actual     
            
            data = Data(
                x=torch.FloatTensor(X_val[i]),
                y=torch.FloatTensor(y_val[i]),
                edge_index = graph_dict[t][0],
                edge_weight= graph_dict[t][1]
            )
            
            val_data_list.append(data)

        # For test data
        test_data_list = []
        for i in range(len(X_test)):
            t           = test_indices[i]  # Get actual 

            data = Data(
                x=torch.FloatTensor(X_test[i]),
                y=torch.FloatTensor(y_test[i]),
                edge_index = graph_dict[t][0],
                edge_weight= graph_dict[t][1]                
            )
            
            test_data_list.append(data)
            
        self.train_dataset  = train_data_list
        self.val_dataset    = val_data_list
        self.test_dataset   = test_data_list