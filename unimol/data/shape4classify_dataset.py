import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
import torch
from .remove_hydrogen_dataset import RemoveHydrogenResiduePocketDataset
from .cropping_dataset import CroppingResiduePocketDataset
from .lmdb_dataset import LMDBDataset
from . import data_utils
import random
from torch_geometric.utils.subgraph import subgraph
import copy
from .key_dataset import KeyDataset
from .distance_dataset import EdgeTypeDataset, DistanceDataset
import pickle
import torch_geometric
from torch_geometric.nn import knn
class CloudDataset(BaseWrapperDataset):
    def __init__(self, dataset, points_num):
        self.dataset = dataset
        self.points_num = points_num

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # sample = torch.cat([sample,torch.zeros([sample.size(0),1])],dim=1)
        num_points = self.points_num
        index = torch.randint(0, sample.size(0), (num_points,))
        # sample[0][-1] = sample.size(0)
        sample = sample[index,:]
        return sample

