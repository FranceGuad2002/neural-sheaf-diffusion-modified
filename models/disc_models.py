# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import torch_sparse

#import Tuple

from torch import nn
from models.sheaf_base import SheafDiffusion
from models import laplacian_builders as lb
from lib import laplace as lap
from models.sheaf_models import LocalConcatSheafLearner, EdgeWeightLearner, LocalConcatSheafLearnerVariant,  OpinionDynamicsFixedSheaf, OpinionDynamicsSheafInitLearner, RotationInvariantSheafLearner


def _apply_gate(L, gate, final_d):
    """Scale cross-node entries of the sparse diffusion Laplacian L by the
    source node's outgoing-message gate; self-block entries (row and col
    belong to the same node) are left at 1.0, so a gated node's own
    self-retention is untouched — only what it sends to *other* nodes is
    suppressed/attenuated. No-op (returns L unchanged) if gate is None, so
    every non-gated forward call is completely unaffected."""
    if gate is None:
        return L
    row, col = L[0]
    node_row, node_col = row // final_d, col // final_d
    is_self_block = node_row == node_col
    entry_gate = torch.where(is_self_block, torch.ones_like(L[1]), gate[node_col])
    return L[0], L[1] * entry_gate


class DiscreteDiagSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(DiscreteDiagSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 0

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()

        """Abbiamo un MLP per ogni layer, ma se non usiamo la non linearità allora ne usiamo solo uno e lo condividiamo tra i layer.
        Questo è perché se non usiamo la non linearità allora il modello è equivalente a un modello con un solo layer di apprendimento della sheaf, quindi non ha senso avere più di un MLP."""

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)

        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.d,), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.d,), sheaf_act=self.sheaf_act))

        
        self.laplacian_builder = lb.DiagLaplacianBuilder(self.graph_size, edge_index, d=self.d,
                                                         normalised=self.normalised,
                                                         deg_normalised=self.deg_normalised,
                                                         add_hp=self.add_hp, add_lp=self.add_lp)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)
        
        # self._last_maps = None
        # self._last_trans_maps = None
        # self._last_laplacian = None

    def forward(self, x, gate=None):

        #print(f"Input x size: {x.detach().cpu().numpy().shape}")

        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)
        #print(f"After initial linear layers x size: {x.detach().cpu().numpy().shape}")

        x0 = x
        self._last_maps = {}
        self._last_trans_maps = {}
        self._last_laplacian = {}
        self._last_node_norms = {}

        for layer in range(self.layers):
            self._last_node_norms[layer] = x.detach().reshape(self.graph_size, -1).norm(dim=1).cpu()
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)

                # print maps to see them
                #print(f"Layer {layer} maps: {maps.detach().cpu().numpy()}, maps length: {maps.size(0)}, edge_index: {self.edge_index}, len edge_index: {self.edge_index.size(1)}")
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)

                self._last_maps[layer] = maps
                self._last_trans_maps[layer] = trans_maps
                self._last_laplacian[layer] = L

                # print(f"Layer {layer} maps: {self._last_maps.detach().cpu().numpy()}")
                # print(f"Layer {layer} Laplacian: {self._last_laplacian[0].detach().cpu().numpy()}, {self._last_laplacian[1].detach().cpu().numpy()}")

            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.graph_size * self.final_d).t()

            if self.right_weights:
                x = self.lin_right_weights[layer](x)

            L_gated = _apply_gate(L, gate, self.final_d)
            x = torch_sparse.spmm(L_gated[0], L_gated[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            coeff = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1))
            x0 = coeff * x0 - x
            x = x0

        # save the norms of node features
        self._last_node_norms[self.layers] = x.detach().reshape(self.graph_size, -1).norm(dim=1).cpu()
        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

class DiscreteBundleSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(DiscreteBundleSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 1
        assert not self.deg_normalised

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))
            
            if self.use_edge_weights:
                self.weight_learners.append(EdgeWeightLearner(self.hidden_dim, edge_index))
        self.laplacian_builder = lb.NormConnectionLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_hp=self.add_hp,
            add_lp=self.add_lp, orth_map=self.orth_trans, normalised=self.normalised)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def get_param_size(self):
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        for weight_learner in self.weight_learners:
            weight_learner.update_edge_index(edge_index)

    def forward(self, x, gate=None):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L = x, None
        self._last_maps = {}
        self._last_trans_maps = {}
        self._last_laplacian = {}
        self._last_node_norms = {}

        for layer in range(self.layers):
            self._last_node_norms[layer] = x.detach().reshape(self.graph_size, -1).norm(dim=1).cpu()
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                x_maps = x_maps.reshape(self.graph_size, -1)
                maps = self.sheaf_learners[layer](x_maps, self.edge_index)
                edge_weights = self.weight_learners[layer](x_maps, self.edge_index) if self.use_edge_weights else None
                L, trans_maps = self.laplacian_builder(maps, edge_weights)
                self.sheaf_learners[layer].set_L(trans_maps)
                self._last_maps[layer] = maps
                self._last_trans_maps[layer] = trans_maps
                self._last_laplacian[layer] = L


            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])

            # Use the adjacency matrix rather than the diagonal
            L_gated = _apply_gate(L, gate, self.final_d)
            x = torch_sparse.spmm(L_gated[0], L_gated[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0
        self._last_node_norms[self.layers] = x.detach().reshape(self.graph_size, -1).norm(dim=1).cpu()
        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class DiscreteGeneralSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(DiscreteGeneralSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 1

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act))
        self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_lp=self.add_lp, add_hp=self.add_hp,
            normalised=self.normalised, deg_normalised=self.deg_normalised)
        self.laplacian_builder_other = lb.GeneralLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_lp=self.add_lp, add_hp=self.add_hp,
            normalised=not self.normalised, deg_normalised=False)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x

    def forward(self, x, gate=None):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L = x, None
        self._last_maps = {}
        self._last_trans_maps = {}
        self._last_laplacian = {}
        self._last_laplacian_other = {}
        self._last_node_norms = {}

        for layer in range(self.layers):
            self._last_node_norms[layer] = x.detach().reshape(self.graph_size, -1).norm(dim=1).cpu()
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)

                self._last_maps[layer] = maps
                self._last_trans_maps[layer] = trans_maps
                self._last_laplacian[layer] = L

                if getattr(self, 'save_other_laplacian', False):
                    with torch.no_grad():
                        L_other, _ = self.laplacian_builder_other(maps)
                    self._last_laplacian_other[layer] = L_other

            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])

            # Use the adjacency matrix rather than the diagonal
            L_gated = _apply_gate(L, gate, self.final_d)
            x = torch_sparse.spmm(L_gated[0], L_gated[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        # To detect the numerical instabilities of SVD.
        assert torch.all(torch.isfinite(x))
        self._last_node_norms[self.layers] = x.detach().reshape(self.graph_size, -1).norm(dim=1).cpu()
        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)





class DiscreteVanillaDiffusion(SheafDiffusion):
    def __init__(self, edge_index, args):
        super(DiscreteVanillaDiffusion, self).__init__(edge_index, args)

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()
        self.layer_norm = nn.ModuleList()
        for i in range(2*self.layers):
            self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
            nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        for i in range(2*self.layers):
            self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
            nn.init.eye_(self.lin_left_weights[-1].weight.data)
        for i in range(self.layers):
            self.layer_norm.append(nn.LayerNorm((self.hidden_channels)))

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        self.sheaf_learners.append(OpinionDynamicsFixedSheaf(
                        self.hidden_channels, self.d, self.edge_index.shape[1]))
        laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_lp=self.add_lp, add_hp=self.add_hp,
            normalised=self.normalised, deg_normalised=self.deg_normalised)
        maps = torch.eye(self.d).reshape(1,self.d,self.d).repeat(self.edge_index.shape[1],1,1).to(self.device)
        self.L, trans_maps = laplacian_builder(maps)
        self.sheaf_learners[0].set_L(trans_maps)
        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1)),requires_grad=args['use_epsilons']))
        #self.lin_alt1 = nn.Linear(self.input_dim,self.hidden_channels+self.d)
        #self.lin_alt2 = nn.Linear(self.d,self.d)
        #self.lin_alt3 = nn.Linear(self.hidden_channels,self.hidden_channels)
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()
        if self.right_weights:
            x = right(x)
        return x

    def forward(self, x, gate=None):
        if self.use_embedding:
            x = F.dropout(x, p=self.input_dropout, training=self.training)

            x = self.lin1(x)
            
            if self.use_act:
                x = F.elu(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        #x1 = self.lin_alt1(x)
        #xd = x1[:,self.hidden_channels:]
        #xd = self.lin_alt2(xd)
        #xf = x1[:,0:self.hidden_channels]
        #xf = self.lin_alt3(xf) 
        #x = torch.bmm(xd.reshape(-1,self.d,1),xf.reshape(-1,1,self.hidden_channels))
        #x = F.elu(x)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0 = x
        self._last_maps = {}
        self._last_trans_maps = {}
        self._last_laplacian = {}
        self._last_node_norms = {}

        for layer in range(self.layers):
            self._last_node_norms[layer] = x.detach().reshape(self.graph_size, -1).norm(dim=1).cpu()
            self._last_laplacian[layer] = self.L
            if layer > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.left_right_linear(x, self.lin_left_weights[2*layer], self.lin_right_weights[2*layer])
            if self.use_act:
                x = F.elu(x)
            x = self.left_right_linear(x, self.lin_left_weights[2*layer+1], self.lin_right_weights[2*layer+1])
            L_gated = _apply_gate(self.L, gate, self.final_d)
            x = torch_sparse.spmm(L_gated[0], L_gated[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x0 = self.layer_norm[layer](x0)
            x = x0

        # To detect the numerical instabilities of SVD.
        assert torch.all(torch.isfinite(x))
        self._last_node_norms[self.layers] = x.detach().reshape(self.graph_size, -1).norm(dim=1).cpu()

        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class DiscreteVanillaDiffusionAlt(SheafDiffusion):
    def __init__(self, edge_index, args):
        super(DiscreteVanillaDiffusionAlt, self).__init__(edge_index, args)
        assert args['d'] > 1


        self.conv_weights = nn.ModuleList()
        self.dim_prod = 1
        for di in self.dim_list:
            self.dim_prod = self.dim_prod*di

        for di in self.dim_list:
            self.conv_weights.append(nn.ModuleList())
            for i in range(self.layers):
                self.conv_weights[-1].append(nn.Linear(self.dim_prod//di, 
                                                       self.dim_prod//di, bias=False))
                nn.init.eye_(self.conv_weights[-1][-1].weight.data)
        self.d = self.dim_list[0]
        self.sheaf_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        self.sheaf_learners.append(OpinionDynamicsFixedSheaf(
                        self.hidden_channels, self.d, self.edge_index.shape[1]))
        laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size, edge_index, d=self.dim_list[0], add_lp=False, add_hp=False,
            normalised=self.normalised, deg_normalised=self.deg_normalised)
        maps = torch.eye(self.d).reshape(1,self.d,self.d).repeat(self.edge_index.shape[1],1,1).to(self.device)
        self.L, trans_maps = laplacian_builder(maps)
        self.sheaf_learners[0].set_L(trans_maps)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.dim_list[0], 1)),requires_grad=args['use_epsilons']))
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        #self.lin_alt1 = nn.Linear(self.input_dim,self.hidden_channels+self.d)
        #self.lin_alt2 = nn.Linear(self.input_dim,self.d)
        #self.lin_alt3 = nn.Linear(self.hidden_channels,self.hidden_channels)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def linear_convs(self, x, layer):
        x = x.reshape([-1]+self.dim_list)
        for i in range(1,len(self.dim_list)):
            x = torch.transpose(x, 1, i+1)
            tshape = x.shape
            x = torch.flatten(x,start_dim = 2)
            x = self.conv_weights[i][layer](x)
            x = x.reshape(tshape)
            x = torch.transpose(x, 1, i+1)
        return x.reshape(self.graph_size*self.dim_list[0],-1)

    def forward(self, x1, gate=None):
        x1 = F.dropout(x1, p=self.input_dropout, training=self.training)
        x1 = self.lin1(x1)
        if self.use_act:
            x1 = F.elu(x1)
        '''
        x1 = self.lin_alt1(x)
        xf = x1[:,0:self.hidden_channels]
        xd = x1[:,self.hidden_channels:]
        x = torch.bmm(xd.reshape(-1,self.d,1),xf.reshape(-1,1,self.hidden_channels))
        x = F.elu(x)
        '''
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        if self.second_linear:
            x1 = self.lin12(x1)
        x = x1.view(self.graph_size * self.dim_list[0], -1)

        x0 = x
        self._last_maps = {}
        self._last_trans_maps = {}
        self._last_laplacian = {}
        self._last_node_norms = {}

        for layer in range(self.layers):
            self._last_node_norms[layer] = x.detach().reshape(self.graph_size, -1).norm(dim=1).cpu()
            self._last_laplacian[layer] = self.L
            x = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)

            x = self.linear_convs(x, layer)

            L_gated = _apply_gate(self.L, gate, self.dim_list[0])
            x = torch_sparse.spmm(L_gated[0], L_gated[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x

            x = x0

        # To detect the numerical instabilities of SVD.
        assert torch.all(torch.isfinite(x))
        self._last_node_norms[self.layers] = x.detach().reshape(self.graph_size, -1).norm(dim=1).cpu()

        x = x.reshape(self.graph_size,-1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class DiscreteJointSheafVanillaDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(DiscreteJointSheafVanillaDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 1

        self.sheaf_learners = nn.ModuleList()
        self.sheaf_learners.append(OpinionDynamicsSheafInitLearner(
                    self.hidden_channels, self.d, self.edge_index.shape[1]))
        self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_lp=False, add_hp=False,
            normalised=self.normalised, deg_normalised=self.deg_normalised)

        self.left_right_idx, _ = lap.compute_left_right_map_index(edge_index)
        self.right_left_idx = torch.cat((self.left_right_idx[1].reshape(1,-1),self.left_right_idx[0].reshape(1,-1)),0)
        sheaf_edge_index_unsorted = torch.cat((self.left_right_idx,self.right_left_idx),1)

        self.dual_laplacian_builder = lb.GeneralLaplacianBuilder(
            edge_index.shape[1],sheaf_edge_index_unsorted, d = self.d,
            normalised=self.normalised, deg_normalised=self.deg_normalised)
        
        self.init_maps = torch.eye(self.d).reshape(1,self.d,self.d).repeat(self.edge_index.shape[1],1,1).to(self.device)
        self.diff_strength = 1
        self.num_steps = 500

    def compute_diffusion_step_of_sheaf(self, x, OldMaps):
        xT = torch.index_select(torch.transpose(x.reshape(self.graph_size,-1,1)[:,0:self.d,:], -2, -1), 
                                dim = 0, index = self.edge_index[0])
        OldMaps = torch.transpose(OldMaps,-1,-2).reshape(-1,self.d)
        
        xTmaps = torch.cat((torch.index_select(xT,0,self.left_right_idx[0]),
                            torch.index_select(xT,0,self.left_right_idx[1])), 0)
        Lsheaf, _ = self.dual_laplacian_builder(xTmaps)
        MapsUpdate = torch.transpose(torch_sparse.spmm(Lsheaf[0], Lsheaf[1], OldMaps.size(0), OldMaps.size(0), OldMaps).reshape((-1,self.d,self.d)),-2,-1)
        MapsUpdated = torch.transpose(OldMaps.reshape(-1,self.d,self.d),-2,-1) - self.diff_strength*MapsUpdate
        return MapsUpdated.reshape(-1,self.d,self.d)

    def forward(self, x, gate=None):
        # NOTE: gate is accepted for API consistency but NOT applied here — this
        # class diffuses the sheaf maps themselves (returns prev_maps[-1], not
        # classification logits) and has no trained checkpoints in this project;
        # the node-block gating math in _apply_gate has not been verified against
        # this class's different x/L structure.
        x = x.reshape(-1,1)
        x0, L = x, None
        prev_maps = [self.init_maps]
        for layer in range(self.num_steps):
            L, trans_maps = self.laplacian_builder(prev_maps[layer])
            self.sheaf_learners[0].set_L(trans_maps)

            #we make the diffusion of the sheaf WITH the same features as the signal diffusion            
            prev_maps.append(self.compute_diffusion_step_of_sheaf(x, prev_maps[layer]))

            # Use the adjacency matrix rather than the diagonal
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)   
                   
            x0 = x0 - x
            x = x0
        # To detect the numerical instabilities of SVD.
        assert torch.all(torch.isfinite(x))

        x = x.reshape(self.graph_size, -1)

        return prev_maps[-1]


class DiscreteJointSheafDiffusionParams(SheafDiffusion):

    def __init__(self, edge_index, args, sheaf_init = []):
        super(DiscreteJointSheafDiffusionParams, self).__init__(edge_index, args)
        assert args['d'] > 1

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()
        self.layer_norm = nn.ModuleList()
        for i in range(2*self.layers):
            self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
            nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        for i in range(2*self.layers):
            self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
            nn.init.eye_(self.lin_left_weights[-1].weight.data)
        for i in range(self.layers):
            self.layer_norm.append(nn.LayerNorm((self.hidden_channels)))

        self.dual_lin_right_weights = nn.ModuleList()
        self.dual_lin_left_weights = nn.ModuleList()

        for i in range(self.layers):
            self.dual_lin_right_weights.append(nn.Linear(self.d, self.d, bias=False))
            nn.init.eye_(self.dual_lin_right_weights[-1].weight.data)
        for i in range(self.layers):
            self.dual_lin_left_weights.append(nn.Linear(self.d, self.d, bias=False))
            nn.init.eye_(self.dual_lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()
        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        
        for i in range(num_sheaf_learners):
            if i == 0 and self.learn_first_maps:
                if self.learnt_map_type == 'diagonal':
                    outShape = (self.d,)
                elif self.learnt_map_type == 'bundle':
                    outShape = (self.get_param_size(),)
                else:  # general
                    outShape = (self.d, self.d)
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=outShape, sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(OpinionDynamicsFixedSheaf(
                        self.hidden_channels, self.d, self.edge_index.shape[1]))

            if i == 0 and self.use_edge_weights:
                self.weight_learners.append(EdgeWeightLearner(self.hidden_dim, edge_index))

        if self.learn_first_maps:
            if self.learnt_map_type == 'diagonal':
                self.first_laplacian_builder = lb.DiagLaplacianBuilder(
                self.graph_size, edge_index, d=self.d, add_hp=self.add_hp,
                add_lp=self.add_lp, normalised=self.normalised,deg_normalised=self.deg_normalised)
            elif self.learnt_map_type == 'bundle':
                self.first_laplacian_builder = lb.NormConnectionLaplacianBuilder(
                self.graph_size, edge_index, d=self.d, add_hp=self.add_hp,
                add_lp=self.add_lp, orth_map=self.orth_trans)
            else:  # general
                self.first_laplacian_builder = lb.GeneralLaplacianBuilder(
                self.graph_size, edge_index, d=self.d, add_hp=self.add_hp,
                add_lp=self.add_lp)

        self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_lp=self.add_lp, add_hp=self.add_hp,
            normalised=self.normalised, deg_normalised=self.deg_normalised)

        #parameters related to the sheaf graph
        self.sheaf_graph_size = edge_index.shape[1]
        self.sheaf_init = sheaf_init
        if self.sheaf_init == []:
            self.sheaf_init = torch.eye(self.d).reshape(1,self.d,self.d).repeat(self.edge_index.shape[1],1,1).to(self.device)

        self.left_right_idx, _ = lap.compute_left_right_map_index(edge_index)
        self.right_left_idx = torch.cat((self.left_right_idx[1].reshape(1,-1),self.left_right_idx[0].reshape(1,-1)),0)
        sheaf_edge_index_unsorted = torch.cat((self.left_right_idx,self.right_left_idx),1)


        self.dual_laplacian_builder = lb.GeneralLaplacianBuilder(
            edge_index.shape[1],sheaf_edge_index_unsorted, d = self.d,
            normalised=args['dual_normalised'],deg_normalised = not args['dual_normalised'],augmented=True)
        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1)), requires_grad=args['use_epsilons']))
        self.dual_epsilons = nn.ParameterList()
        for i in range(self.layers-1):
            self.dual_epsilons.append(nn.Parameter(torch.zeros((self.d, 1)), requires_grad=args['use_epsilons']))
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def get_param_size(self):
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x
    
    def dual_left_right_linear(self,x,layer):
        if self.dual_left_linear:
            x = x.t().reshape(-1, self.d)
            x = self.dual_lin_left_weights[layer](x)
            x = x.reshape(-1, self.edge_index.shape[1] * self.d).t()
        if self.dual_right_linear:
            x = self.dual_lin_right_weights[layer](x)
        return x

    def compute_diffusion_step_of_sheaf(self, x, OldMaps, layer):
        if self.diff_strength == 0:
            return OldMaps
        xT = torch.index_select(torch.transpose(x.reshape(self.graph_size,
                                                          -1,
                                                          self.hidden_channels)[:,0:self.d,:], 
                                                -2, 
                                                -1), 
                                dim = 0, index = self.edge_index[0])
        OldMaps = torch.transpose(OldMaps,-1,-2).reshape(-1,self.d)
        
        xTmaps = torch.cat((torch.index_select(xT,0,self.left_right_idx[0]),
                            torch.index_select(xT,0,self.left_right_idx[1])), 0)
        Lsheaf, _ = self.dual_laplacian_builder(xTmaps)
        MapsUpdate = torch.transpose(torch_sparse.spmm(Lsheaf[0], 
                                                         Lsheaf[1], 
                                                         OldMaps.size(0), 
                                                         OldMaps.size(0), 
                                                         OldMaps).reshape((-1,
                                                                           self.d,
                                                                           self.d)),-2,-1)
        if self.deg_normalised:
            MapsUpdate = 2*MapsUpdate
        if not self.dual_linear:
            MapsUpdate = torch.tanh(MapsUpdate)
        MapsUpdated = torch.transpose(((1 + torch.tanh(self.dual_epsilons[layer])).tile(self.edge_index.shape[1], 1)*OldMaps).reshape(-1,self.d,self.d),-2,-1) - self.diff_strength*MapsUpdate
        return MapsUpdated.reshape(-1,self.d,self.d)

    def forward(self, x, gate=None):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L = x, None
        self._last_maps = {}
        self._last_trans_maps = {}
        self._last_laplacian = {}
        self._last_node_norms = {}

        prev_maps = [self.sheaf_init]
        for layer in range(self.layers):
            self._last_node_norms[layer] = x.detach().reshape(self.graph_size, -1).norm(dim=1).cpu()
            if layer == 0 and self.learn_first_maps:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                x_maps = x_maps.reshape(self.graph_size,-1)
                maps = (self.sheaf_learners[layer](x_maps, self.edge_index))
                #edge_weights = self.weight_learners[layer](x_maps, self.edge_index) if self.use_edge_weights else None
                L, trans_maps = self.first_laplacian_builder(maps)
                if self.learnt_map_type == 'diagonal':
                    prev_maps[layer] = torch.diag_embed(maps)
                elif self.learnt_map_type == 'bundle':
                    prev_maps[layer] = self.first_laplacian_builder.orth_transform(maps)
                else:  # general
                    prev_maps[layer] = maps
            else :
                L, trans_maps = self.laplacian_builder(prev_maps[layer])

            self.sheaf_learners[layer].set_L(trans_maps)
            self._last_maps[layer] = prev_maps[layer].detach()
            self._last_trans_maps[layer] = trans_maps
            self._last_laplacian[layer] = L

            x = F.dropout(x, p=self.dropout, training=self.training)

            prev_maps[layer] = self.dual_left_right_linear(prev_maps[layer].reshape(-1,self.d),layer).reshape(-1,self.d,self.d)

            #we make the diffusion of the sheaf WITH the same features as the signal diffusion
            if layer < self.layers-1:
                prev_maps.append(self.compute_diffusion_step_of_sheaf(x, prev_maps[layer],layer))

            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[2*layer](x)
                x = x.reshape(-1, self.graph_size * self.final_d).t()

            if self.right_weights:
                x = self.lin_right_weights[2*layer](x)

            if self.use_act:
                x = F.elu(x)

            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[2*layer+1](x)
                x = x.reshape(-1, self.graph_size * self.final_d).t()

            if self.right_weights:
                x = self.lin_right_weights[2*layer+1](x)

            L_gated = _apply_gate(L, gate, self.final_d)
            x = torch_sparse.spmm(L_gated[0], L_gated[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)


            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x0 = self.layer_norm[layer](x0)
            x = x0
            
        # To detect the numerical instabilities of SVD.
        assert torch.all(torch.isfinite(x))
        self._last_node_norms[self.layers] = x.detach().reshape(self.graph_size, -1).norm(dim=1).cpu()

        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class DiscreteJointSheafDiffusionParamsAlt(SheafDiffusion):

    def __init__(self, edge_index, args, sheaf_init = []):
        super(DiscreteJointSheafDiffusionParamsAlt, self).__init__(edge_index, args)
        assert args['d'] > 1

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.dual_lin_right_weights = nn.ModuleList()
        self.dual_lin_left_weights = nn.ModuleList()

        for i in range(self.layers):
            self.dual_lin_right_weights.append(nn.Linear(self.d, self.d, bias=False))
            nn.init.eye_(self.dual_lin_right_weights[-1].weight.data)
        for i in range(self.layers):
            self.dual_lin_left_weights.append(nn.Linear(self.d, self.d, bias=False))
            nn.init.eye_(self.dual_lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()
        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        
        for i in range(num_sheaf_learners):
            if i == 0 and self.learn_first_maps:
                if self.learnt_map_type == 'diagonal':
                    outShape = (self.d,)
                elif self.learnt_map_type == 'bundle':
                    outShape = (self.get_param_size(),)
                else:  # general
                    outShape = (self.d, self.d)
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=outShape, sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(OpinionDynamicsFixedSheaf(
                        self.hidden_channels, self.d, self.edge_index.shape[1]))

            if i == 0 and self.use_edge_weights:
                self.weight_learners.append(EdgeWeightLearner(self.hidden_dim, edge_index))

        if self.learn_first_maps:
            if self.learnt_map_type == 'diagonal':
                self.first_laplacian_builder = lb.DiagLaplacianBuilder(
                self.graph_size, edge_index, d=self.d, add_hp=self.add_hp,
                add_lp=self.add_lp, normalised=self.normalised,deg_normalised=self.deg_normalised)
            elif self.learnt_map_type == 'bundle':
                self.first_laplacian_builder = lb.NormConnectionLaplacianBuilder(
                self.graph_size, edge_index, d=self.d, add_hp=self.add_hp,
                add_lp=self.add_lp, orth_map=self.orth_trans)
            else:  # general
                self.first_laplacian_builder = lb.GeneralLaplacianBuilder(
                self.graph_size, edge_index, d=self.d, add_hp=self.add_hp,
                add_lp=self.add_lp)

        self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_lp=self.add_lp, add_hp=self.add_hp,
            normalised=self.normalised, deg_normalised=self.deg_normalised)

        #parameters related to the sheaf graph
        self.sheaf_graph_size = edge_index.shape[1]
        self.sheaf_init = sheaf_init
        if self.sheaf_init == []:
            self.sheaf_init = torch.eye(self.d).reshape(1,self.d,self.d).repeat(self.graph_size,1,1).to(self.device)

        self.left_right_idx, _ = lap.compute_left_right_map_index(edge_index)
        self.right_left_idx = torch.cat((self.left_right_idx[1].reshape(1,-1),self.left_right_idx[0].reshape(1,-1)),0)
        sheaf_edge_index_unsorted = torch.cat((self.left_right_idx,self.right_left_idx),1)


        self.dual_laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size,edge_index, d = self.d,
            normalised=args['dual_normalised'],deg_normalised = not args['dual_normalised'],augmented=True)
        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1)), requires_grad=args['use_epsilons']))
        self.dual_epsilons = nn.ParameterList()
        for i in range(self.layers-1):
            self.dual_epsilons.append(nn.Parameter(torch.zeros((self.d, 1)), requires_grad=args['use_epsilons']))
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.lin11 = nn.Linear(self.input_dim,self.d*self.d)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def get_param_size(self):
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x
    
    def dual_left_right_linear(self,x,layer):
        if self.dual_left_linear:
            x = x.t().reshape(-1, self.d)
            x = self.dual_lin_left_weights[layer](x)
            x = x.reshape(-1, self.graph_size * self.d).t()
        if self.dual_right_linear:
            x = self.dual_lin_right_weights[layer](x)
        return x

    def compute_diffusion_step_of_sheaf(self, x, OldMaps, layer):
        if self.diff_strength == 0:
            return OldMaps
        
        xT = torch.index_select(torch.transpose(x.reshape(self.graph_size,
                                                          -1,
                                                          self.hidden_channels)[:,0:self.d,:], 
                                                -2, 
                                                -1), 
                                dim = 0, index = self.edge_index[0])
        OldMaps = torch.transpose(OldMaps,-1,-2).reshape(-1,self.d)
        
        xTmaps = torch.cat((torch.index_select(xT,0,self.left_right_idx[0]),
                            torch.index_select(xT,0,self.left_right_idx[1])), 0)
        Lsheaf, _ = self.dual_laplacian_builder(xTmaps)
        MapsUpdate = torch.transpose(torch_sparse.spmm(Lsheaf[0], 
                                                         Lsheaf[1], 
                                                         OldMaps.size(0), 
                                                         OldMaps.size(0), 
                                                         OldMaps).reshape((-1,
                                                                           self.d,
                                                                           self.d)),-2,-1)
        if self.deg_normalised:
            MapsUpdate = 2*MapsUpdate
        if not self.dual_linear:
            MapsUpdate = torch.tanh(MapsUpdate)
        MapsUpdated = torch.transpose(((1 + torch.tanh(self.dual_epsilons[layer])).tile(self.graph_size, 1)*OldMaps).reshape(-1,self.d,self.d),-2,-1) - self.diff_strength*MapsUpdate
        return MapsUpdated.reshape(-1,self.d,self.d)

    def forward(self, x, gate=None):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L = x, None
        self._last_maps = {}
        self._last_trans_maps = {}
        self._last_laplacian = {}
        self._last_node_norms = {}

        prev_maps = [self.sheaf_init]
        for layer in range(self.layers):
            self._last_node_norms[layer] = x.detach().reshape(self.graph_size, -1).norm(dim=1).cpu()
            Lmaps = torch.index_select(prev_maps[layer],dim=0,index = self.edge_index[0])
            Lmaps = torch.cat((torch.index_select(Lmaps,0,self.left_right_idx[0]),
                            torch.index_select(Lmaps,0,self.left_right_idx[1])), 0)
            L, trans_maps = self.laplacian_builder(Lmaps)

            self.sheaf_learners[layer].set_L(trans_maps)
            self._last_maps[layer] = prev_maps[layer].detach()
            self._last_trans_maps[layer] = trans_maps
            self._last_laplacian[layer] = L

            x = F.dropout(x, p=self.dropout, training=self.training)

            prev_maps[layer] = self.dual_left_right_linear(prev_maps[layer].reshape(-1,self.d),layer).reshape(-1,self.d,self.d)

            #we make the diffusion of the sheaf WITH the same features as the signal diffusion
            if layer < self.layers-1:
                prev_maps.append(self.compute_diffusion_step_of_sheaf(x, prev_maps[layer],layer))

            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.graph_size * self.final_d).t()

            if self.right_weights:
                x = self.lin_right_weights[layer](x)
            # Use the adjacency matrix rather than the diagonal
            L_gated = _apply_gate(L, gate, self.final_d)
            x = torch_sparse.spmm(L_gated[0], L_gated[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0
            
        # To detect the numerical instabilities of SVD.
        assert torch.all(torch.isfinite(x))
        self._last_node_norms[self.layers] = x.detach().reshape(self.graph_size, -1).norm(dim=1).cpu()

        x = x.reshape(self.graph_size, -1)
        #x = torch.cat((x,prev_maps[-1].reshape(self.graph_size,-1)),-1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)