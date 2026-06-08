# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
import os
import os.path as osp
import torch_geometric.transforms as T
import pandas as pd

from typing import Optional, Callable, List, Union
from torch_sparse import SparseTensor, coalesce
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.utils import remove_self_loops
from utils.classic import Planetoid
from definitions import ROOT_DIR



class Actor(InMemoryDataset):
    r"""
    Code adapted from https://github.com/pyg-team/pytorch_geometric/blob/2.0.4/torch_geometric/datasets/actor.py

    The actor-only induced subgraph of the film-director-actor-writer
    network used in the
    `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
    Each node corresponds to an actor, and the edge between two nodes denotes
    co-occurrence on the same Wikipedia page.
    Node features correspond to some keywords in the Wikipedia pages.
    The task is to classify the nodes into five categories in term of words of
    actor's Wikipedia.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt'
                ] + [f'film_split_0.6_0.2_{i}.npz' for i in range(10)]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        for f in self.raw_file_names[:2]:
            download_url(f'{self.url}/new_data/film/{f}', self.raw_dir)
        for f in self.raw_file_names[2:]:
            download_url(f'{self.url}/splits/{f}', self.raw_dir)

    def process(self):

        with open(self.raw_paths[0], 'r') as f:
            data = [x.split('\t') for x in f.read().split('\n')[1:-1]]

            rows, cols = [], []
            for n_id, col, _ in data:
                col = [int(x) for x in col.split(',')]
                rows += [int(n_id)] * len(col)
                cols += col
            x = SparseTensor(row=torch.tensor(rows), col=torch.tensor(cols))
            x = x.to_dense()

            y = torch.empty(len(data), dtype=torch.long)
            for n_id, _, label in data:
                y[int(n_id)] = int(label)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            # Remove self-loops
            edge_index, _ = remove_self_loops(edge_index)
            # Make the graph undirected
            edge_index = to_undirected(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        train_masks, val_masks, test_masks = [], [], []
        for f in self.raw_paths[2:]:
            tmp = np.load(f)
            train_masks += [torch.from_numpy(tmp['train_mask']).to(torch.bool)]
            val_masks += [torch.from_numpy(tmp['val_mask']).to(torch.bool)]
            test_masks += [torch.from_numpy(tmp['test_mask']).to(torch.bool)]
        train_mask = torch.stack(train_masks, dim=1)
        val_mask = torch.stack(val_masks, dim=1)
        test_mask = torch.stack(test_masks, dim=1)

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])


class WikipediaNetwork(InMemoryDataset):
    r"""
    Code adapted from https://github.com/pyg-team/pytorch_geometric/blob/2.0.4/torch_geometric/datasets/wikipedia_network.py

    The Wikipedia networks introduced in the
    `"Multi-scale Attributed Node Embedding"
    <https://arxiv.org/abs/1909.13021>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features represent several informative nouns in the Wikipedia pages.
    The task is to predict the average daily traffic of the web page.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"chameleon"`,
            :obj:`"crocodile"`, :obj:`"squirrel"`).
        geom_gcn_preprocess (bool): If set to :obj:`True`, will load the
            pre-processed data as introduced in the `"Geom-GCN: Geometric
            Graph Convolutional Networks" <https://arxiv.org/abs/2002.05287>_`,
            in which the average monthly traffic of the web page is converted
            into five categories to predict.
            If set to :obj:`True`, the dataset :obj:`"crocodile"` is not
            available.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    """

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        assert self.name in ['chameleon', 'squirrel']
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str]]:
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
        x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
        x = torch.tensor(x, dtype=torch.float)
        y = [int(r.split('\t')[2]) for r in data]
        y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
        edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
        # Remove self-loops
        edge_index, _ = remove_self_loops(edge_index)
        # Make the graph undirected
        edge_index = to_undirected(edge_index)
        edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])


class WebKB(InMemoryDataset):
    r"""
    Code adapted from https://github.com/pyg-team/pytorch_geometric/blob/2.0.4/torch_geometric/datasets/webkb.py

    The WebKB datasets used in the
    `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features are the bag-of-words representation of web pages.
    The task is to classify the nodes into one of the five categories, student,
    project, course, staff, and faculty.
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cornell"`,
            :obj:`"Texas"` :obj:`"Washington"`, :obj:`"Wisconsin"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/'
           '1c4c04f93fa6ada91976cda8d7577eec0e3e5cce/new_data')

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['cornell', 'texas', 'washington', 'wisconsin']

        super(WebKB, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{self.name}/{name}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float32)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index = to_undirected(edge_index)
            # We also remove self-loops in these datasets in order not to mess up with the Laplacian.
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


def get_fixed_splits(data, dataset_name, seed):
    with np.load(f'splits/{dataset_name}_split_0.6_0.2_{seed}.npz') as splits_file:
        train_mask = splits_file['train_mask']
        val_mask = splits_file['val_mask']
        test_mask = splits_file['test_mask']

    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
    data.test_mask = torch.tensor(test_mask, dtype=torch.bool)

    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        data.train_mask[data.non_valid_samples] = False
        data.test_mask[data.non_valid_samples] = False
        data.val_mask[data.non_valid_samples] = False
        print("Non zero masks", torch.count_nonzero(data.train_mask + data.val_mask + data.test_mask))
        print("Nodes", data.x.size(0))
        print("Non valid", len(data.non_valid_samples))
    else:
        assert torch.count_nonzero(data.train_mask + data.val_mask + data.test_mask) == data.x.size(0)
    return data



class SyntheticData(InMemoryDataset):
    
    def __init__(self, root, name, args, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['synthetic_exp']
        self.num_nodes = args.num_nodes
        self.n_classes = args.num_classes
        self.num_feats = args.num_feats
        self.het = args.het_coef
        self.p = 1-args.edge_noise
        self.K = args.node_degree
        self.r = args.ellipsoid_radius
        self.feat_noise = args.feat_noise
        self.just_add_noise = args.just_add_noise
        self.ellipsoids = args.ellipsoids
        self.matriu_corr = None
        if args.classes_corr is not None:
            n_classes = int(np.sqrt(len(args.classes_corr)))
            self.matriu_corr = np.zeros((n_classes,n_classes))
            for i in range(n_classes):
                for j in range(n_classes):
                    self.matriu_corr[i][j] = args.classes_corr[i*n_classes+j]
        else:
            self.matriu_corr = self.het*(1/(self.n_classes-1))*np.ones((self.n_classes,self.n_classes))
            for i in range(self.n_classes):
                self.matriu_corr[i][i] = 1-self.het
        super(SyntheticData, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return 'synthetic_data.pt'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def generate_features(self):
        y = torch.ones((self.num_nodes))
        for i in range(self.num_nodes):
            y[i] = np.random.randint(0,self.n_classes)
        y = y.type(torch.LongTensor)
        if self.ellipsoids == True:
            class_params = torch.tensor(np.random.rand(self.n_classes,self.num_feats))
            mean_class_params = torch.mean(class_params,dim=0)
            torch.save(class_params,self.raw_dir+'/class_params.pt')
            x = torch.ones((self.num_nodes,self.num_feats),dtype=float)
            x_coords = np.pi*np.random.rand(self.num_nodes,self.num_feats-1)
            x_coords[:][-1] = x_coords[:][-1]*2
            noise = torch.normal(mean=torch.zeros(self.num_nodes,self.num_feats,dtype=float),std=self.feat_noise)
            for i in range(self.num_nodes):
                x[i] = class_params[y[i]]*self.r*x[i]
                #we also rescale the noise to make it a percentage
                noise[i] = class_params[y[i]]*noise[i]
                for j in range(self.num_feats-1):
                    x[i][j] = np.cos(x_coords[i][j])*x[i][j]
                    for k in range(j+1,self.num_feats):
                        x[i][k] = np.sin(x_coords[i][j])*x[i][k]
            x = x+noise
        else:
            y = torch.ones((self.num_nodes))
            for i in range(self.num_nodes):
                y[i] = np.random.randint(0,self.n_classes)
            y = torch.tensor(y).detach()
            y = y.type(torch.LongTensor)
            #generate random multivariate normal distributions
            center_means = torch.tensor(np.random.rand(self.num_feats),dtype=float)
            means = torch.tensor(np.random.rand(self.n_classes,self.num_feats),dtype=float)*0.25
            cov_mat = torch.tensor(np.random.rand(self.num_feats,self.num_feats),dtype=float)
            cov_mat = torch.matmul(torch.transpose(cov_mat,-2,-1),cov_mat)
            for i in range(self.n_classes):
                means[i] = center_means+means[i]
                

            x = torch.ones((self.num_nodes,self.num_feats))

            for i in range(x.shape[0]):
                act_class = y[i]
                distrib = torch.distributions.multivariate_normal.MultivariateNormal(means[act_class], cov_mat)
                x[i] = distrib.rsample()
        x = x.type(torch.FloatTensor)
        return x, y
    
    def generate_edges(self, y):

        no_edge = [[] for l in range(self.num_nodes)]
        
        for i in range(self.num_nodes):
            no_edge[i] = [ [] for l in range(self.n_classes)]
            for j in range(self.num_nodes):
                if 0 == abs(i-j)%self.num_nodes or abs(i-j) > self.K/2:
                    no_edge[i][y[j]].append(j)
        edge_index = [[],[]]
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if 0 < (j-i)%self.num_nodes and (j-i)%self.num_nodes <= self.K/2 and np.random.random() <= self.p:
                    ck = int(np.random.choice(range(self.n_classes),1,p=self.matriu_corr[y[i]]))
                    while len(no_edge[i][ck]) == 0:
                        ck = (ck+1)%self.n_classes
                    ind_k = np.random.randint(0,len(no_edge[i][ck]))
                    k = no_edge[i][ck][ind_k]
                    edge_index[0] = edge_index[0]+[i,k]
                    edge_index[1] = edge_index[1]+[k,i]
                    no_edge[i][ck].pop(ind_k)
                elif 0 < (j-i)%self.num_nodes and (j-i)%self.num_nodes <= self.K/2:
                    edge_index[0] = edge_index[0]+[i,j]
                    edge_index[1] = edge_index[1]+[j,i]
        return edge_index
    
    def download(self):
        matriu_corr = self.matriu_corr
        #we try to read from data file the features, for fair comparison
        try:
            x = torch.load(self.raw_dir+'/x_data.pt')
            y = torch.load(self.raw_dir+'/y_data.pt')
            if self.just_add_noise:
                class_params = torch.load(self.raw_dir+'/class_params.pt')
                mean_class_params = torch.mean(class_params,dim=0)
                noise = torch.normal(mean=torch.zeros(self.num_nodes,
                                                      self.num_feats,
                                                      dtype=float),
                                     std=self.feat_noise)
                for i in range(self.num_nodes):
                    noise[i] = mean_class_params*noise[i]
                x = x+noise
                x = x.type(torch.FloatTensor)
        except:
            x, y = self.generate_features()
            torch.save(x,self.raw_dir+'/x_data.pt')
            torch.save(y,self.raw_dir+'/y_data.pt')
        try:
            edge_index = torch.load(self.raw_dir+'/edge_data.pt')
        except:
            edge_index = self.generate_edges(y)
            torch.save(edge_index,self.raw_dir+'/edge_data.pt')
        data = [Data(x=x,edge_index = torch.tensor(edge_index),y = y)]
        torch.save(self.collate(data),self.raw_dir+'/synthetic_data.pt')

    def process(self):
        data = torch.load(self.raw_dir+'/synthetic_data.pt')
        x = data[0].x
        y = data[0].y

        edge_index = data[0].edge_index
        # We also remove self-loops in these datasets in order not to mess up with the Laplacian.
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


def get_fixed_splits(data, dataset_name, seed):
    try:
        with np.load(f'splits/{dataset_name}_split_0.6_0.2_{seed}.npz') as splits_file:
            train_mask = splits_file['train_mask']
            val_mask = splits_file['val_mask']
            test_mask = splits_file['test_mask']
    except:
        n = data.x.shape[0]
        masks = torch.randperm(n)
        train_index = int(0.6*n)
        val_index = train_index + int(0.2*n)
        train_mask = np.full(n, False)
        val_mask = np.full(n,False)
        test_mask = np.full(n,False)
        if dataset_name == 'ogbn-arxiv':
            train_idx = pd.read_csv('datasets/ogbn_arxiv/split/time/train.csv.gz')
            train_idx = list(train_idx[train_idx.columns[0]])
            val_idx = pd.read_csv('datasets/ogbn_arxiv/split/time/valid.csv.gz')
            val_idx = list(val_idx[val_idx.columns[0]])
            test_idx = pd.read_csv('datasets/ogbn_arxiv/split/time/test.csv.gz')
            test_idx = list(test_idx[test_idx.columns[0]])
            train_mask[train_idx] = True
            val_mask[val_idx] = True
            test_mask[test_idx] = True
        else:
            train_mask[masks[0:train_index]] = True
            val_mask[masks[train_index:val_index]] = True
            test_mask[masks[val_index:]] = True
        np.savez(f'splits/{dataset_name}_split_0.6_0.2_{seed}.npz',train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
    data.test_mask = torch.tensor(test_mask, dtype=torch.bool)

    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        data.train_mask[data.non_valid_samples] = False
        data.test_mask[data.non_valid_samples] = False
        data.val_mask[data.non_valid_samples] = False
        print("Non zero masks", torch.count_nonzero(data.train_mask + data.val_mask + data.test_mask))
        print("Nodes", data.x.size(0))
        print("Non valid", len(data.non_valid_samples))
    elif dataset_name != 'ogbn-arxiv':
        assert torch.count_nonzero(data.train_mask + data.val_mask + data.test_mask) == data.x.size(0)
    return data



def get_dataset(name, args=None):
    data_root = osp.join(ROOT_DIR, 'datasets')
    if name in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(root=data_root, name=name, transform=T.NormalizeFeatures())
    elif name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root=data_root, name=name, transform=T.NormalizeFeatures())
    elif name == 'film':
        dataset = Actor(root=data_root, transform=T.NormalizeFeatures())
    elif name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=data_root, name=name, transform=T.NormalizeFeatures())
    elif name == "synthetic_exp":
        dataset = SyntheticData(data_root,name, args)
    elif name == 'ogbn-arxiv':
        dataset = torch.load(data_root+'/ogbn_arxiv/processed/geometric_data_processed.pt')
        edge_index = dataset[0].edge_index 
        edge_index,_ = remove_self_loops(edge_index)
        # Make the graph undirected
        edge_index = to_undirected(edge_index)
        edge_index, _ = coalesce(edge_index, None, dataset[0].x.size(0), dataset[0].x.size(0))
        dataset[0].edge_index = edge_index
        dataset[0].y = dataset[0].y.reshape(-1)
    elif name == 'roman_empire':
        edge_index = torch.tensor(np.transpose(np.load(data_root+'/roman_empire/raw/edges.npy')))
        edge_index2 = torch.cat([edge_index[1].reshape(1,-1),edge_index[0].reshape(1,-1)], dim=0)
        edge_index = torch.cat((edge_index,edge_index2),dim=1)
        dataset = [Data(x=torch.tensor(np.load(data_root+'/roman_empire/raw/node_features.npy')),
                     edge_index = edge_index,
                     y = torch.tensor(np.load(data_root+'/roman_empire/raw/node_labels.npy')))]
    elif name == 'minesweeper':
        edge_index = torch.tensor(np.transpose(np.load(data_root+'/minesweeper/raw/edges.npy')))
        edge_index2 = torch.cat([edge_index[1].reshape(1,-1),edge_index[0].reshape(1,-1)], dim=0)
        edge_index = torch.cat((edge_index,edge_index2),dim=1)
        dataset = [Data(x=torch.tensor(np.load(data_root+'/minesweeper/raw/node_features.npy')),
                     edge_index = edge_index,
                     y = torch.tensor(np.load(data_root+'/minesweeper/raw/node_labels.npy')))]
    elif name in ['amazon_ratings', 'tolokers', 'questions']:
        edge_index = torch.tensor(np.transpose(np.load(data_root+f'/{name}/raw/edges.npy')))
        edge_index2 = torch.cat([edge_index[1].reshape(1,-1),edge_index[0].reshape(1,-1)], dim=0)
        edge_index = torch.cat((edge_index,edge_index2),dim=1)
        dataset = [Data(x=torch.tensor(np.load(data_root+f'/{name}/raw/node_features.npy')),
                     edge_index = edge_index,
                     y = torch.tensor(np.load(data_root+f'/{name}/raw/node_labels.npy')))]
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')
    return dataset