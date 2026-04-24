# system
import os

# standard data science libraries
import numpy as np
import pandas as pd

# plotting
import seaborn as sns
import networkx as nx
import matplotlib as mpl
from matplotlib import axes
import matplotlib.pyplot as plt

# core machine learning library
import torch

# utilities
from typing import Literal, List

# various datasets imports
# parameters for data retrieval and model training
stalk_dim = 1
hidden_channels = 16
dataset_list = ["texas", "cornell", "wisconsin", "cora"]
layers = 2
epochs = 500
fold = 0

# storing learned maps and laplacians from the experiments

dataset_general_dict = {dataset: {} for dataset in dataset_list}

# basic loop to load the maps and laplacians for each dataset and layer, given the same hyper-parameters (stalk_dim, hidden_channels, layers, epochs, fold)
for dataset in dataset_list:
    maps, laplacians = [], []
    for layer in range(layers):
    # load maps and laplacian for each provided layer
        maps_path = f"../results/maps/{dataset}/stalk_dim-{stalk_dim}/{layers}-layers/{hidden_channels}-hidden/{epochs}-epochs/DiagSheaf_{dataset}_layer{layer}_fold{fold}_seed43.pt"
        laplacian_path = f"../results/laplacians/{dataset}/stalk_dim-{stalk_dim}/{layers}-layers/{hidden_channels}-hidden/{epochs}-epochs/DiagSheaf_{dataset}_layer{layer}_fold{fold}_seed43.pt"
        maps.append(torch.load(maps_path))
        laplacians.append(torch.load(laplacian_path))
    dataset_general_dict[dataset]["maps"] = maps
    dataset_general_dict[dataset]["laplacians"] = laplacians  

# modified maps storing
for dataset in dataset_list:
    maps_dfs = []
    for map_item in dataset_general_dict[dataset]["maps"]:
        map_df = pd.DataFrame(map_item.cpu().tolist(), columns=["source", "target", "map_value"])
        map_df[["source", "target"]] = map_df[["source", "target"]].astype(int)
        map_df.sort_values(by=["source", "target"], ascending=[True, True], inplace=True)
        maps_dfs.append(map_df)
    dataset_general_dict[dataset]["maps_dataframes"] = maps_dfs

# modified laplacians storing
for dataset in dataset_list:
    laplacians_dfs = []
    for laplacian_item in dataset_general_dict[dataset]["laplacians"]:
        # print(f"Loading laplacian for {dataset}: {laplacian_item}")
        laplacian_df = pd.DataFrame(laplacian.cpu().tolist())
        laplacians_dfs.append(laplacian_df)
    dataset_general_dict[dataset]["laplacians_dataframes"] = laplacians_dfs

# normalized and unnormalized laplacians storing
for dataset in dataset_list:
    
    maps_dfs = dataset_general_dict[dataset]["maps_dataframes"]
    laplacians_dfs = dataset_general_dict[dataset]["laplacians_dataframes"]
    
    l0_list_paper, l0_list = [], []

    # for 0-Laplacian as in the paper, we reconstruct the matrix by pivoting our current df
    for lap in laplacians_dfs:
        df = lap.transpose().copy()
        df.rename(columns={0: "source", 1: "target", 2: "value"}, inplace=True)
        df_new = df.pivot(index="source", columns="target", values="value").fillna(0)
        l0 = df_new.to_numpy()
        l0_list_paper.append(l0)

    # for 0-Laplacian computed from scratch, we need maps matrix manipulation
    for map_df in maps_dfs:
        
        df = map_df.copy()
        df[["source", "target"]] = df[["source", "target"]].astype(int)
        df["signed_map"] = np.where(df["target"] < df["source"], df["map_value"], -df["map_value"])

        df["edge"] = df.apply(lambda r: (min(r["source"], r["target"]), max(r["source"], r["target"])), axis=1)
        
        nodes = sorted(set(df["source"]).union(set(df["target"])))
        edges = sorted(set(df["edge"]))

        B = df.pivot(index="source", columns="edge", values="signed_map").fillna(0).to_numpy()
        # B_pd = df.pivot(index="source", columns="edge", values="signed_map").fillna(0)
        
        l0 = B @ B.T
        l0_list.append(l0)

    dataset_general_dict[dataset]["l0_list_norm"] = l0_list_paper
    dataset_general_dict[dataset]["l0_list_unnorm"] = l0_list

    # updated function for edge weights and curvatures - slightly different, takes into account different dataframes selections
def forman_weber_curvature_general(
        L: Literal["paper_Laplacian", "computed_Laplacian"] = "computed_Laplacian",
        dataset: str = "texas",
        layer_selection: int = 0,
        node_weight_mode: Literal[
            "units",
            "standard_nodes_degrees", "standard_nodes_degrees_inverted",
            "standard_nodes_degrees_augmented", "standard_nodes_degrees_augmented_inverted",
            "sheaf_nodes_degrees", "sheaf_nodes_degrees_inverted",
            "sheaf_nodes_degrees_augmented", "sheaf_nodes_degrees_augmented_inverted"
            ] = "units",
        output_weights_mode: bool = False):
    
    """
    Compute the Forman-Weber curvature for a given Laplacian and incidence matrix.
    Inputs:
    - L: Literal["paper Laplacian", "computed Laplacian"], whether to use the 0-Laplacian as provided in the paper or the one computed from the maps-derived B matrices (default: "computed Laplacian")
    - dataset: str, the dataset to consider for curvature computation (default: "texas")
    - layer_selection: int, the layer to consider for curvature computation (default: 0)
    - node_weight_mode: the mode for computing node weights (default: "units")
    - output_weights_mode: whether to output just the computed node and edge weights along with the starting Laplacian (default: False)

    Outputs:

    if output_weights_mode is True (default):
    - L_0: the selected Laplacian matrix
    - edge_weights_dict: dictionary w/ edge (input, output)-tuple keys, edge weight values
    - node_weights: the computed node weights
    - edge_weights: the computed edge weights

    else:
    - L_0: the selected Laplacian matrix
    - curvature_dict: dictionary w/ edge (input, output)-tuple keys, curvature values
    - node_weights: the computed node weights
    - edge_weights: the computed edge weights
    """
    
    if L == "paper_Laplacian":
        L_0 = dataset_general_dict[dataset]["l0_list_norm"][layer_selection]
        print(L_0)

    elif L == "computed_Laplacian":
        L_0 = dataset_general_dict[dataset]["l0_list_unnorm"][layer_selection]
        print(L_0)

    else:
        raise ValueError(f"Invalid L value: {L}. Expected one of 'paper_Laplacian', 'computed_Laplacian'.")
        
    # 1. edge-weights
    edge_weights = np.abs(L_0)

    # 2. node-weights
    if node_weight_mode == "units":
        node_weights = np.ones(L_0.shape[0])

    elif node_weight_mode == "standard_nodes_degrees":
        D = np.diag(np.count_nonzero(L_0, axis=1))
        node_weights = np.diag(D)

    elif node_weight_mode == "standard_nodes_degrees_inverted":
        D = np.diag(np.count_nonzero(L_0, axis=1))
        node_weights = np.diag(np.linalg.inv(D))

    elif node_weight_mode == "standard_nodes_degrees_augmented":
        D = np.diag(np.count_nonzero(L_0, axis=1))
        D = D + np.eye(D.shape[0])  # Add identity matrix for augmentation
        node_weights = np.diag(D)

    elif node_weight_mode == "standard_nodes_degrees_augmented_inverted":
        D = np.diag(np.count_nonzero(L_0, axis=1))
        D = D + np.eye(D.shape[0])  # Add identity matrix for augmentation
        node_weights = np.diag(np.linalg.inv(D))

    elif node_weight_mode == "sheaf_nodes_degrees":
        D = np.diag(L_0)
        node_weights = np.diag(D)

    elif node_weight_mode == "sheaf_nodes_degrees_inverted":
        D = np.diag(np.diag(L_0))
        node_weights = np.diag(np.linalg.inv(D))
    
    elif node_weight_mode == "sheaf_nodes_degrees_augmented":
        D = np.diag(L_0)
        D = D + np.eye(D.shape[0])  # Add identity matrix for augmentation
        node_weights = np.diag(D)
    
    elif node_weight_mode == "sheaf_nodes_degrees_augmented_inverted":
        D = np.diag(L_0)
        D = D + np.eye(D.shape[0])  # Add identity matrix for augmentation

        node_weights = np.diag(np.linalg.inv(D))

    else:
        raise ValueError(f"Invalid node_weight_mode value: {node_weight_mode}")

    if output_weights_mode:
        
        edge_weights_dict = {(i, j): edge_weights[i, j] for i in range(L_0.shape[0]) for j in range(L_0.shape[1]) if (i != j and L_0[i, j] != 0) and (i < j)}
        
        return L_0, edge_weights_dict, node_weights, edge_weights
    
    else:
        # 3. curvature computation
        curvature_dict = {}
        for i in range(L_0.shape[0]):
            for j in range(L_0.shape[1]):
                if (i != j and L_0[i, j] != 0) and (i < j):
                    edge_key = (i, j)

                    # first term: weight(i) / weight(i,j)
                    term1 = node_weights[i] / edge_weights[i, j]

                    # second term: weight(j) / weight(i,j)
                    term2 = node_weights[j] / edge_weights[i, j]

                    # third term: sum_{over edges k incident to i} of weight of i / sqrt(weight(i,j) * weight(i,k))
                    term3 = 0
                    for k in range(L_0.shape[1]):
                        if k != j and L_0[i,k] != 0:
                            if i < k:
                                term3 += node_weights[i] / np.sqrt(edge_weights[i, j] * edge_weights[i, k])
                            elif i > k:
                                term3 += node_weights[i] / np.sqrt(edge_weights[i, j] * edge_weights[k, i])
                    
                    # fourth term: sum_{over edges k incident to j} of weight of j / sqrt(weight(i,j) * weight(j,k))
                    term4 = 0
                    for k in range(L_0.shape[1]):
                        if k != i and L_0[j,k] != 0:
                            if j < k:
                                term4 += node_weights[j] / np.sqrt(edge_weights[i, j] * edge_weights[j, k])
                            elif j > k:
                                term4 += node_weights[j] / np.sqrt(edge_weights[i, j] * edge_weights[k, j])

                    # summing up the terms and multiplying by the edge weight
                    # curvature_dict[edge_key] = (term1 + term2 - term3 - term4)
                    curvature_dict[edge_key] = edge_weights[i, j] * (term1 + term2 - term3 - term4)
        
        return L_0, curvature_dict, node_weights, edge_weights
    
# plotting, for each dataset, the edge weights, subdividing:
# - by layer
# - by Laplacian selection (normalized vs unnormalized)
# - by visualization type (graph visualisation vs distribution comparison)

for dataset in dataset_list:
    
    unnorm_edge_weights_layer_dict = {layer_selection: forman_weber_curvature_general(L="computed_Laplacian", dataset=dataset, layer_selection=layer_selection, output_weights_mode=True)[1] for layer_selection in range(layers)}
    norm_edge_weights_layer_dict = {layer_selection: forman_weber_curvature_general(L="paper_Laplacian", dataset=dataset, layer_selection=layer_selection, output_weights_mode=True)[1] for layer_selection in range(layers)}

    fig, axes = plt.subplots(3, layers, figsize=(6 * layers, 4 * layers))
    # fig, axes = plt.subplots(3, layers)

    for layer in range(layers):
        
        # 1. unnormalized edge weights
        edge_weights_dict = unnorm_edge_weights_layer_dict[layer]
        
        G = nx.Graph()
        G.add_nodes_from(list(range(L_0.shape[0])))
        G.add_edges_from(sorted(edge_weights_dict.keys()))

        pos = nx.kamada_kawai_layout(G)

        edges = list(G.edges())
        edge_colors = [
            edge_weights_dict[tuple(sorted(edge))] for edge in edges
        ]

        vmin = np.min(edge_colors)
        vmax = np.max(edge_colors)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.coolwarm

        nx.draw_networkx_nodes(G, pos=pos, node_color='black', node_size=1, alpha=0.6, ax=axes[0, layer])
        nx.draw_networkx_edges(G, pos=pos, edgelist=edges, edge_color=edge_colors, edge_cmap=cmap, edge_vmin=vmin, edge_vmax=vmax, width=1, ax=axes[0, layer])

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = plt.colorbar(sm, ax=axes[0, layer])
        # cbar.set_label("Edge weights")

        axes[0, layer].set_title(f"Layer {layer + 1} - unnormalized")
        axes[0, layer].axis("off")

        # 2. normalized edge weights
        edge_weights_dict = norm_edge_weights_layer_dict[layer]

        G = nx.Graph()
        G.add_nodes_from(list(range(L_0.shape[0])))
        G.add_edges_from(sorted(edge_weights_dict.keys()))

        pos = nx.kamada_kawai_layout(G)
        edges = list(G.edges())
        edge_colors = [
            edge_weights_dict[tuple(sorted(edge))] for edge in edges
        ]

        vmin = np.min(edge_colors)
        vmax = np.max(edge_colors)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.coolwarm

        nx.draw_networkx_nodes(G, pos=pos, node_color='black', node_size=1, alpha=0.6, ax=axes[2, layer])
        nx.draw_networkx_edges(G, pos=pos, edgelist=edges, edge_color=edge_colors, edge_cmap=cmap, edge_vmin=vmin, edge_vmax=vmax, width=1, ax=axes[2 , layer])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axes[2, layer])
        # cbar.set_label("Edge weights")
        axes[2, layer].set_title(f"Layer {layer + 1} - normalized")
        axes[2, layer].axis("off")

        # 3. unnormalized vs normalized edge weights distribution comparison
        # unnormalized edge weights
        edge_weights_dict = unnorm_edge_weights_layer_dict[layer]
        edge_weight_values_unnorm = list(edge_weights_dict.values())
        # normalized edge weights
        edge_weights_dict = norm_edge_weights_layer_dict[layer]
        edge_weight_values_norm = list(edge_weights_dict.values())
        sns.kdeplot(edge_weight_values_unnorm, label="Unnormalized", ax=axes[1, layer])
        sns.kdeplot(edge_weight_values_norm, label="Normalized", ax=axes[1, layer])
        # axes[1, layer].set_title(f"Layer {layer + 1}")
        # axes[1, layer].set_xlabel("Edge weights")
        # axes[1, layer].set_ylabel("Density")
        axes[1, layer].legend()
    
    plt.suptitle(f"Edge weights across layers - {dataset.capitalize()}", fontsize=19)
    plt.tight_layout()
    plt.show()    