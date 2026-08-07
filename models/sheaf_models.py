# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import numpy as np

from typing import Tuple
from abc import abstractmethod
from torch import nn
from lib import laplace as lap


class SheafLearner(nn.Module):
    """Base model that learns a sheaf from the features and the graph structure."""
    def __init__(self):
        super(SheafLearner, self).__init__()
        self.L = None

    @abstractmethod
    def forward(self, x, edge_index):
        raise NotImplementedError()

    def set_L(self, weights):
        self.L = weights.clone().detach()


class LocalConcatSheafLearner(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(self, in_channels: int, out_shape: Tuple[int, ...], sheaf_act="tanh"):
        super(LocalConcatSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.linear1 = torch.nn.Linear(in_channels*2, int(np.prod(out_shape)), bias=False)

        if sheaf_act == 'id':
            self.act = lambda x: x
        elif sheaf_act == 'tanh':
            self.act = torch.tanh
        elif sheaf_act == 'elu':
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(self, x, edge_index):
        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        maps = self.linear1(torch.cat([x_row, x_col], dim=1))
        maps = self.act(maps)

        #print(maps,edge_index)

        # sign = maps.sign()
        # maps = maps.abs().clamp(0.05, 1.0) * sign
        # BE CAREFUL, THIS IS ROW-MAJOR ORDER, NOT COLUMN-MAJOR
        # [a11, a12, a21, a22] -> [[a11, a21], [a12, a22]]
        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])


class LocalNodeSheafLearner(SheafLearner):
    """Learns one map per node from that node's own features, rather than one map per edge from
    a concatenated pair. Used to build a per-vertex map O_v shared by all of v's incident edges
    (the 'flat' Bochner sheaf), so it takes a single node feature matrix, not an edge_index pair.
    """

    def __init__(self, in_channels: int, out_shape: Tuple[int, ...], sheaf_act="tanh"):
        super(LocalNodeSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.linear1 = torch.nn.Linear(in_channels, int(np.prod(out_shape)), bias=False)

        if sheaf_act == 'id':
            self.act = lambda x: x
        elif sheaf_act == 'tanh':
            self.act = torch.tanh
        elif sheaf_act == 'elu':
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(self, x):
        maps = self.act(self.linear1(x))

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])


class LocalConcatSheafLearnerVariant(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(self, d: int, hidden_channels: int, out_shape: Tuple[int, ...], sheaf_act="tanh"):
        super(LocalConcatSheafLearnerVariant, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.d = d
        self.hidden_channels = hidden_channels
        self.linear1 = torch.nn.Linear(hidden_channels * 2, int(np.prod(out_shape)), bias=False)
        # self.linear2 = torch.nn.Linear(self.d, 1, bias=False)

        # std1 = 1.414 * math.sqrt(2. / (hidden_channels * 2 + 1))
        # std2 = 1.414 * math.sqrt(2. / (d + 1))
        #
        # nn.init.normal_(self.linear1.weight, 0.0, std1)
        # nn.init.normal_(self.linear2.weight, 0.0, std2)

        if sheaf_act == 'id':
            self.act = lambda x: x
        elif sheaf_act == 'tanh':
            self.act = torch.tanh
        elif sheaf_act == 'elu':
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(self, x, edge_index):

        """Seleziona le righe corrispondenti a row e col da x, concatena le caratteristiche e passa attraverso un layer lineare e un'attivazione."""

        row, col = edge_index

        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        x_cat = torch.cat([x_row, x_col], dim=-1)
        x_cat = x_cat.reshape(-1, self.d, self.hidden_channels * 2).sum(dim=1)

        x_cat = self.linear1(x_cat)

        # x_cat = x_cat.t().reshape(-1, self.d)
        # x_cat = self.linear2(x_cat)
        # x_cat = x_cat.reshape(-1, edge_index.size(1)).t()

        maps = self.act(x_cat)

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])


class AttentionSheafLearner(SheafLearner):

    """Utilizza l'attention per costruire le mappe di restrizione.
    Concatena le caratteristiche dei nodi, le passa attraverso un layer lineare
    e poi applica softmax per ottenere le mappe di restrizione."""

    def __init__(self, in_channels, d):
        super(AttentionSheafLearner, self).__init__()
        self.d = d
        self.linear1 = torch.nn.Linear(in_channels*2, d**2, bias=False)

    def forward(self, x, edge_index):
        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        maps = self.linear1(torch.cat([x_row, x_col], dim=1)).view(-1, self.d, self.d)

        id = torch.eye(self.d, device=edge_index.device, dtype=maps.dtype).unsqueeze(0)
        return id - torch.softmax(maps, dim=-1)


class EdgeWeightLearner(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(self, in_channels: int, edge_index):
        super(EdgeWeightLearner, self).__init__()
        self.in_channels = in_channels
        self.linear1 = torch.nn.Linear(in_channels*2, 1, bias=False)
        self.full_left_right_idx, _ = lap.compute_left_right_map_index(edge_index, full_matrix=True)

    def forward(self, x, edge_index):
        _, full_right_idx = self.full_left_right_idx

        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        weights = self.linear1(torch.cat([x_row, x_col], dim=1))
        weights = torch.sigmoid(weights)

        edge_weights = weights * torch.index_select(weights, index=full_right_idx, dim=0)
        return edge_weights

    def update_edge_index(self, edge_index):
        self.full_left_right_idx, _ = lap.compute_left_right_map_index(edge_index, full_matrix=True)


class DiagEdgeWeightLearner(SheafLearner):
    """Learns a per-channel diagonal edge scale Sigma_e = scale * sigmoid(linear(h_u + h_v)) in
    (0, scale)^d. Sigma_{(u,v)} == Sigma_{(v,u)} holds automatically since h_u + h_v is symmetric
    in (u, v) -- no reverse-edge bookkeeping needed.
    """

    def __init__(self, in_channels: int, d: int, scale: float = 1.0):
        super(DiagEdgeWeightLearner, self).__init__()
        self.in_channels = in_channels
        self.d = d
        self.scale = scale
        self.linear1 = torch.nn.Linear(in_channels, d, bias=False)

    def forward(self, x, edge_index):
        row, col = edge_index
        h_sum = torch.index_select(x, dim=0, index=row) + torch.index_select(x, dim=0, index=col)
        return self.scale * torch.sigmoid(self.linear1(h_sum))


class QuadraticFormSheafLearner(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(self, in_channels: int, out_shape: Tuple[int]):
        super(QuadraticFormSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape

        tensor = torch.eye(in_channels).unsqueeze(0).tile(int(np.prod(out_shape)), 1, 1)
        self.tensor = nn.Parameter(tensor)

    def forward(self, x, edge_index):
        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        maps = self.map_builder(torch.cat([x_row, x_col], dim=1))

        if len(self.out_shape) == 2:
            return torch.tanh(maps).view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return torch.tanh(maps).view(-1, self.out_shape[0])




class OpinionDynamicsSheafInitLearner(SheafLearner):
    def __init__(self, hidden_channels, d, num_maps):
        super(OpinionDynamicsSheafInitLearner, self).__init__()
        self.d = d
        self.hidden_channels = hidden_channels
        #initialize maps as identity matrices
        #self.maps = torch.eye(self.d).reshape(1,self.d,self.d).repeat(num_maps,1,1)
        self.maps = nn.Parameter(torch.eye(self.d).reshape(1,self.d,self.d).repeat(num_maps,1,1),requires_grad=True)

    def forward(self, x, edge_index, maps, diff_strength = 1):
        
        if diff_strength == 0:
            res_maps = maps.clone().detach()
            return res_maps
        row = edge_index[0]
        col = edge_index[1]
        #we reshape the vector to have graph size x dimension x hidden channels
        #and change the order of the dimensions to easily acces each vector of
        #each channel and each node
        x2 = x.reshape(-1,self.d,self.hidden_channels)
        self.left_right_idx, self.vertex_tril_idx = lap.compute_left_right_map_index(edge_index)
        left_idx, right_idx = self.left_right_idx
        #update the sheaf with the data
        #select all the node vectors of the left maps
        xu = torch.index_select(x2, dim=0, index=row)
        xu = torch.index_select(xu, dim = 0, index = left_idx)
        #select all the node vectors of the right maps
        xv = torch.index_select(x2, dim=0, index=col)
        xv = torch.index_select(xv, dim = 0, index = left_idx)
        res_maps = maps.clone().detach()
        #select all the maps
        left_maps = torch.index_select(res_maps, index=left_idx, dim=0)
        right_maps = torch.index_select(res_maps, index=right_idx, dim=0)

        #update the maps according to the ODE
        left_maps = left_maps - diff_strength*torch.bmm((torch.bmm(left_maps,xu)-torch.bmm(right_maps,xv)),torch.transpose(xu,-2,-1))
        right_maps = right_maps - diff_strength*torch.bmm((torch.bmm(right_maps,xv)-torch.bmm(left_maps,xu)),torch.transpose(xv,-2,-1))

        #save changes
        res_maps[left_idx] = left_maps
        res_maps[right_idx] = right_maps
        return res_maps



class OpinionDynamicsFixedSheaf(SheafLearner):
    def __init__(self, hidden_channels, d, num_maps):
        super(OpinionDynamicsFixedSheaf, self).__init__()
        self.d = d
        self.hidden_channels = hidden_channels

    def forward(self, x, edge_index, maps, diff_strength = 1):
        if diff_strength == 0:
            res_maps = maps.clone().detach()
            return res_maps
        row, col = edge_index
        
        #we want to select the left and right maps
        self.left_right_idx, self.vertex_tril_idx = lap.compute_left_right_map_index(edge_index)
        left_idx, right_idx = self.left_right_idx
        #update the sheaf with the data
        #select all the node vectors of the left maps
        xu = torch.index_select(x, dim=0, index=row)
        xu = torch.index_select(xu, dim = 0, index = left_idx)
        #select all the node vectors of the right maps
        xv = torch.index_select(x, dim=0, index=col)
        #here it's left_idx because selecting index=col we are already selecting the other edge
        xv = torch.index_select(xv, dim = 0, index = left_idx)
        res_maps = maps.clone().detach()
        #select all the maps
        left_maps = torch.index_select(res_maps, index=left_idx, dim=0)
        right_maps = torch.index_select(res_maps, index=right_idx, dim=0)

        #update the maps according to the ODE
        left_maps = left_maps - diff_strength*torch.bmm((torch.bmm(left_maps,xu)-torch.bmm(right_maps,xv)),torch.transpose(xu,-2,-1))
        right_maps = right_maps - diff_strength*torch.bmm((torch.bmm(right_maps,xv)-torch.bmm(left_maps,xu)),torch.transpose(xv,-2,-1))

        #save changes
        res_maps[left_idx] = left_maps
        res_maps[right_idx] = right_maps
        return res_maps


class RotationInvariantSheafLearner(SheafLearner):
    def __init__(self, d: int, hidden_channels: int, edge_index, graph_size, out_shape: Tuple[int, ...], time_dep: bool, transform = None, 
                 sheaf_act="tanh"):
        super(RotationInvariantSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.d = d
        self.hidden_channels = hidden_channels
        self.linear1 = torch.nn.Linear(d*d, int(np.prod(out_shape)), bias=True)

        self.time_dep = time_dep
        self.left_right_idx, _ = lap.compute_left_right_map_index(edge_index)
        right_left_idx = torch.cat((self.left_right_idx[1].reshape(1,-1),self.left_right_idx[0].reshape(1,-1)),0)
        sheaf_edge_index_unsorted = torch.cat((self.left_right_idx,right_left_idx),1)
        self.graph_size = graph_size
        self.dual_laplacian_builder = lb.GeneralLaplacianBuilder(
            edge_index.shape[1],sheaf_edge_index_unsorted, d = self.d,
            normalised=False,deg_normalised = True, augmented=False)
        
        self.transform = transform

        if sheaf_act == 'id':
            self.act = lambda x: x
        elif sheaf_act == 'tanh':
            self.act = torch.tanh
        elif sheaf_act == 'elu':
            self.act = F.elu

    def forward(self, x, edge_index, Maps):

        if Maps == None or not self.time_dep:
            Maps2 = torch.eye(self.d).reshape(1,self.d,self.d).repeat(edge_index.shape[1],1,1).to(x.device)
        elif self.transform is not None:
            if self.transform == torch.diag:
                Maps2 = torch.stack([self.transform(map1) for map1 in Maps])
            else:
                Maps2 = self.transform(Maps)
        else:
            Maps2 = Maps
        xT = torch.index_select(torch.transpose(x.reshape(self.graph_size,-1,self.hidden_channels)[:,0:self.d,:], -2, -1), 
                                dim = 0, index = edge_index[0])
        OldMaps = torch.transpose(Maps2,-1,-2).reshape(-1,self.d)
        
        xTmaps = torch.cat((torch.index_select(xT,0,self.left_right_idx[0]),
                            torch.index_select(xT,0,self.left_right_idx[1])), 0)
        Lsheaf, _ = self.dual_laplacian_builder(xTmaps)
        node_edge_sims = 2*torch.transpose(torch_sparse.spmm(Lsheaf[0], Lsheaf[1], OldMaps.size(0), OldMaps.size(0), OldMaps).reshape((-1,self.d,self.d)),-2,-1)

        node_edge_sims = self.linear1(node_edge_sims.reshape(-1,self.d*self.d))
        maps = self.act(node_edge_sims)
        if len(self.out_shape) == 2:
            maps = maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            maps = maps.view(-1, self.out_shape[0])
        return maps    


class AttentionSheafLearner(SheafLearner):

    def __init__(self, in_channels, d):
        super(AttentionSheafLearner, self).__init__()
        self.d = d
        self.linear1 = torch.nn.Linear(in_channels*2, d**2, bias=False)

    def forward(self, x, edge_index):
        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        maps = self.linear1(torch.cat([x_row, x_col], dim=1)).view(-1, self.d, self.d)

        id = torch.eye(self.d, device=edge_index.device, dtype=maps.dtype).unsqueeze(0)
        return id - torch.softmax(maps, dim=-1)


class EdgeWeightLearner(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(self, in_channels: int, edge_index):
        super(EdgeWeightLearner, self).__init__()
        self.in_channels = in_channels
        self.linear1 = torch.nn.Linear(in_channels*2, 1, bias=False)
        self.full_left_right_idx, _ = lap.compute_left_right_map_index(edge_index, full_matrix=True)

    def forward(self, x, edge_index):
        _, full_right_idx = self.full_left_right_idx

        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        weights = self.linear1(torch.cat([x_row, x_col], dim=1))
        weights = torch.sigmoid(weights)

        edge_weights = weights * torch.index_select(weights, index=full_right_idx, dim=0)
        return edge_weights

    def update_edge_index(self, edge_index):
        self.full_left_right_idx, _ = lap.compute_left_right_map_index(edge_index, full_matrix=True)


class QuadraticFormSheafLearner(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(self, in_channels: int, out_shape: Tuple[int]):
        super(QuadraticFormSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape

        tensor = torch.eye(in_channels).unsqueeze(0).tile(int(np.prod(out_shape)), 1, 1)
        self.tensor = nn.Parameter(tensor)

    def forward(self, x, edge_index):
        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        maps = self.map_builder(torch.cat([x_row, x_col], dim=1))

        if len(self.out_shape) == 2:
            return torch.tanh(maps).view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return torch.tanh(maps).view(-1, self.out_shape[0])

