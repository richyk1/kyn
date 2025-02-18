import torch
from torch_geometric.data import Data
from typing import Optional, Union, List, Dict, Any, Literal
from collections import defaultdict

import GraphWithMetadata


def from_rustworkx(
    graph: GraphWithMetadata,
    group_node_attrs: Optional[Union[List[str], Literal["all"]]] = None,
    group_edge_attrs: Optional[Union[List[str], Literal["all"]]] = None,
) -> Data:
    # Step 1: Create node index mapping
    original_nodes = graph.node_indices()
    index_map = {orig: i for i, orig in enumerate(original_nodes)}

    # Step 2: Initialize data dictionary
    data_dict: Dict[str, Any] = defaultdict(list)

    # Step 3: Process node attributes
    node_data = [graph.get_node_data(idx) for idx in original_nodes]
    node_attrs = []
    if node_data:
        if not all(isinstance(d, dict) for d in node_data):
            raise TypeError("All node data must be dictionaries")
        node_attrs = list(node_data[0].keys())
        for data in node_data[1:]:
            if data.keys() != node_data[0].keys():
                raise ValueError("Inconsistent node attributes")

        for feat_dict in node_data:
            for key, value in feat_dict.items():
                data_dict[key].append(value)

    # Step 4: Process edges and edge attributes
    edges = graph.edge_list()
    edge_index = torch.empty((2, len(edges)), dtype=torch.long)
    if edges:
        sources, targets = zip(*edges)
        edge_data = [graph.get_edge_data(s, t) for s, t in edges]
        mapped_sources = [index_map[s] for s in sources]
        mapped_targets = [index_map[t] for t in targets]
        edge_index = torch.tensor([mapped_sources, mapped_targets], dtype=torch.long)
    data_dict["edge_index"] = edge_index

    edge_attrs = []
    if edges and edge_data:
        if not all(isinstance(d, dict) for d in edge_data):
            raise TypeError("All edge data must be dictionaries")
        edge_attrs = list(edge_data[0].keys())
        for data in edge_data[1:]:
            if data.keys() != edge_data[0].keys():
                raise ValueError("Inconsistent edge attributes")

        for feat_dict in edge_data:
            for key, value in feat_dict.items():
                # Handle attribute name conflicts
                new_key = f"edge_{key}" if key in node_attrs else key
                data_dict[new_key].append(value)

    # Step 5: Process graph attributes
    graph_attrs = getattr(graph, "metadata", {})
    for key, value in graph_attrs.items():
        new_key = f"graph_{key}" if key in node_attrs else key
        data_dict[new_key] = value

    # Step 6: Convert lists to tensors
    for key, value in data_dict.items():
        if isinstance(value, list):
            try:
                data_dict[key] = torch.tensor(value, dtype=torch.float)
            except Exception as e:
                raise ValueError(f"Failed to convert {key} to tensor: {str(e)}")

    # Step 7: Create Data object
    data = Data.from_dict(data_dict)

    # Step 8: Handle attribute grouping
    if group_node_attrs == "all":
        group_node_attrs = node_attrs
    elif group_node_attrs is None:
        group_node_attrs = []

    if group_edge_attrs == "all":
        group_edge_attrs = edge_attrs
    elif group_edge_attrs is None:
        group_edge_attrs = []

    # Process node attribute grouping
    if group_node_attrs:
        xs = []
        for key in group_node_attrs:
            if key not in data:
                raise KeyError(f"Node attribute {key} not found")
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.x = torch.cat(xs, dim=-1)

    # Process edge attribute grouping
    if group_edge_attrs:
        edge_attrs = []
        for key in group_edge_attrs:
            # Handle potential renamed edge attributes
            edge_key = f"edge_{key}" if key in node_attrs else key
            if edge_key not in data:
                raise KeyError(f"Edge attribute {key} not found")
            x = data[edge_key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            edge_attrs.append(x)
            del data[edge_key]
        data.edge_attr = torch.cat(edge_attrs, dim=-1)

    # Handle empty node features
    if data.x is None and data.pos is None:
        data.num_nodes = len(original_nodes)

    return data
