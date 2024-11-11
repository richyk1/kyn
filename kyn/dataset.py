import glob
import json
import pickle
import random
from pathlib import Path

import networkx as nx
import torch
from networkx.readwrite import json_graph
from torch_geometric.utils import from_networkx
from tqdm import tqdm

from loguru import logger


class KYNDataset:
    def __init__(
        self,
        root_data_path: str,
        dataset_naming_convetion: str,
        filter_strs: list[str] = [],
        sample_size: int = -1,
        exclude: bool = False,
        with_edge_features: bool = True,
    ):
        logger.info("Starting KYN dataset init")
        self.root_data_path = root_data_path

        logger.info(f"Scanning {self.root_data_path} for filepaths")
        self.file_paths = self._load_file_paths()
        if len(self.file_paths) == 0:
            raise ValueError("No files found within path provided.")
        self.dataset_len = len(self.file_paths)
        logger.info(f"{self.dataset_len} filepaths found")

        if filter_strs:
            logger.info("Starting filtering...")
            self.filter_strs = filter_strs
            self.file_paths = self._filter(exclude)
            self.dataset_len = len(self.file_paths)
            logger.info(f"Filtering complete. Remaining files: {self.dataset_len}")

        if sample_size:
            if sample_size != -1:
                self.target_sample_size = sample_size
                self.sample_size = int(sample_size * 1.05)
                if self.sample_size > self.dataset_len:
                    logger.warning(
                        "The scaled sample size to deal with failures is bigger than the dataset length. This will introduce duplicates."
                    )
            else:
                self.target_sample_size = None
                self.sample_size = sample_size

            logger.info(f"Sample size of {self.sample_size} provided.")
            self.file_paths = self._sample()
            self.dataset_len = len(self.file_paths)
            logger.info(f"Remaining files: {self.dataset_len}")

        if dataset_naming_convetion not in ["cisco", "binkit", "trex", "binarycorp"]:
            raise ValueError(
                "The dataset naming convetion only has four options - 'cisco', 'binkit', 'trex' or 'binarycorp'"
            )
        else:
            self.dataset_naming_convetion = dataset_naming_convetion

        self.with_edge_features = with_edge_features
        self.binary_func_id_index = []
        self.labels = []
        self.graphs = []
        self.no_graphs_failed = 0

    def _load_file_paths(self):
        """
        Load all the files within a given target root data path

        :return: A list of filepaths
        """
        return glob.glob(f"{self.root_data_path}/**/**.json")

    def _sample(self):
        """
        Randomly sample the filepaths found if a sample size has been provided
        else provide original dataset.

        :return: A list of filepaths of size self.sample_size otherwise return the original list
        """
        if self.sample_size > self.dataset_len:
            raise ValueError(
                f"Sample size {self.sample_size} bigger than the dataset length {self.dataset_len}"
            )

        if self.sample_size != -1:
            return random.choices(self.file_paths, k=self.sample_size)
        else:
            self.sample_size = len(self.file_paths)
            return self.file_paths

    def _filter(self, exclude: bool):
        if exclude:
            return [
                filepath
                for filepath in self.file_paths
                if any(filter_str not in filepath for filter_str in self.filter_strs)
            ]
        else:
            return [
                filepath
                for filepath in self.file_paths
                if any(filter_str in filepath for filter_str in self.filter_strs)
            ]

    def load_and_transform_graphs(self):
        for file_path in tqdm(self.file_paths):
            binary_function_id = self.get_binary_func_id(file_path)

            if binary_function_id not in self.binary_func_id_index:
                self.binary_func_id_index.append(binary_function_id)

            try:
                # Load the graph into a JSON object
                data = json.load(open(file_path))

                # Iterate each node within the JSON to remove un-needed data
                for node in data["nodes"]:
                    self.clean_graph_nodes(node)

                # Load JSON into a Neworkx DiGraph Object
                G = json_graph.adjacency_graph(data)

                if len(list(G.edges())) == 0:
                    self.no_graphs_failed += 1
                    continue

                # Add edge-betweenness values to each of the edges if set (True by default)
                if self.with_edge_features:
                    bb = nx.edge_betweenness_centrality(G, normalized=False)
                    nx.set_edge_attributes(G, bb, "weight")

                # Generate an integer label based on the index of the binary_function_id
                label = self._generate_graph_label(G, binary_function_id)

                # Add function name to the graph metadata
                G.graph["name"] = Path(file_path).name.split("-")[0]

                if self.with_edge_features:
                    pyg = from_networkx(
                        G,
                        group_node_attrs=[
                            "ninstrs",
                            "edges",
                            "indegree",
                            "outdegree",
                            "nlocals",
                            "nargs",
                        ],
                        group_edge_attrs=["weight"],
                    )
                else:
                    pyg = from_networkx(
                        G,
                        group_node_attrs=[
                            "ninstrs",
                            "edges",
                            "indegree",
                            "outdegree",
                            "nlocals",
                            "nargs",
                        ],
                    )
                pyg.x = pyg.x.to(torch.float32)
                # drop processed graphs with no edges
                pyg.validate()
                self.graphs.append(pyg)

            except Exception as e:
                print(f"Failed to load {file_path} - Exception: {e}")

        if self.target_sample_size is None:
            logger.info(
                f"Processed {self.sample_size} graphs. {self.no_graphs_failed} failed to process resulting in {self.sample_size - self.no_graphs_failed} saved."
            )
        else:
            self.graphs = self.graphs[: self.target_sample_size]
            self.labels = self.labels[: self.target_sample_size]
            logger.info(
                f"Processed {self.sample_size} graphs. {self.target_sample_size} were generated after {self.no_graphs_failed} failed from {self.sample_size} processed"
            )

    def get_binary_func_id(self, file_path: str) -> str:
        if self.dataset_naming_convetion == "binkit":
            return self.get_binkit_binary_func_id(file_path)
        elif self.dataset_naming_convetion == "binarycorp":
            return self.get_binarycorp_binary_func_id(file_path)
        elif self.dataset_naming_convetion == "trex":
            return self.get_trex_binary_func_id(file_path)
        else:
            return self.get_cisco_talos_binary_func_id(file_path)

    def get_binkit_binary_func_id(self, file_path: str) -> str:
        return (
            Path(file_path).parent.name.split("_")[-2]
            + Path(file_path).name.split("-")[0]
        )

    def get_cisco_talos_binary_func_id(self, file_path: str) -> str:
        return (
            Path(file_path).parent.name.split("-")[-3].split("_")[1]
            + Path(file_path).name.split("-")[0]
        )

    def get_trex_binary_func_id(self, file_path: str) -> str:
        return (
            Path(file_path).parent.name.split("_")[2]
            + Path(file_path).name.split("-")[0]
        )

    def get_binarycorp_binary_func_id(self, file_path: str) -> str:
        return (
            Path(file_path).parent.name.split("-")[-5]
            + Path(file_path).name.split("-")[0]
        )

    def clean_graph_nodes(self, node):
        del node["functionFeatureSubset"]["signature"]
        del node["functionFeatureSubset"]["name"]
        for k, v in node["functionFeatureSubset"].items():
            node[k] = v
        del node["functionFeatureSubset"]

    def save_dataset(self, save_prefix: str):
        with open(f"{save_prefix}-graphs.pickle", "wb") as fp:
            pickle.dump(self.graphs, fp)
            fp.close()

        with open(f"{save_prefix}-labels.pickle", "wb") as fp:
            pickle.dump(self.labels, fp)
            fp.close()

    def _generate_graph_label(self, G: nx.DiGraph, binary_function_id: str) -> int:
        """
        Generate the label for a function call graphlet

        :param G: the networkx graph to generate metadata for
        :param binary_function_id: the binary name + function id string for the graph
        :return: integer label for the graph
        """
        label = self.binary_func_id_index.index(binary_function_id)
        G.graph["label"] = label
        self.labels.append(label)

        return label
