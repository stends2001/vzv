import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# Pytorch
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric_temporal.signal import temporal_signal_split, StaticGraphTemporalSignal
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader

# Specifics
import networkx as nx
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Union, Tuple, Dict

# General
import sys
import os
from pathlib import Path


n_dirs_back     = 0
current_dir     = Path(os.getcwd())
src_dir         = str(current_dir.parents[n_dirs_back])
project_root    = str(current_dir.parents[n_dirs_back+1])

if src_dir not in sys.path:
    sys.path.append(src_dir)

from lib import *