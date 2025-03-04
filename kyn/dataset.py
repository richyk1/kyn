import glob
import orjson
import pickle
import random
from pathlib import Path

import rustworkx as rx
from rustworkx import PyDiGraph
from tqdm import tqdm
from collections import Counter

from loguru import logger
from GraphWithMetadata import GraphWithMetadata
from from_rustworkx import from_rustworkx


class KYNDataset:
    def __init__(
        self,
        root_data_path: str,
        dataset_naming_convention: str,
        filter_strs: list[str] = [],
        sample_size: int = -1,
        exclude: bool = False,
        with_edge_features: bool = True,
        min_examples_per_class: int = 2,
    ):
        logger.info("Starting KYN dataset init")
        self.root_data_path = root_data_path
        self.min_examples_per_class = min_examples_per_class

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

        if dataset_naming_convention not in [
            "cisco",
            "binkit",
            "trex",
            "binarycorp",
            "custom",
        ]:
            raise ValueError(
                "The dataset naming convetion only has four options - 'cisco', 'binkit', 'trex' or 'binarycorp'"
            )
        else:
            self.dataset_naming_convention = dataset_naming_convention

        self.with_edge_features = with_edge_features
        self.binary_func_id_index = []
        self.labels = []
        self.graphs = []
        self.no_graphs_failed = 0
        self.avg_examples_per_class = 0.0

        self.binary_function_ids = [
            self.get_binary_func_id(fp) for fp in self.file_paths
        ]
        self.id_counts = Counter(self.binary_function_ids)

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

        self.binary_function_ids = [
            self.get_binary_func_id(fp) for fp in self.file_paths
        ]
        self.id_counts = Counter(self.binary_function_ids)
        valid_ids = {id_ for id_, count in self.id_counts.items() if count >= 2}
        self.file_paths = [
            fp for fp in self.file_paths if self.get_binary_func_id(fp) in valid_ids
        ]

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
            # Get frequency-based weights [functions appearing in more binaries get higher weight]
            weights = [self.id_counts[id] for id in self.binary_function_ids]

            # Normalize weights and sample
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]

            return random.choices(
                self.file_paths, weights=normalized_weights, k=self.sample_size
            )
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
        sampled_id_counts = Counter()
        for file_path in tqdm(self.file_paths):
            binary_function_id = self.get_binary_func_id(file_path)
            sampled_id_counts[binary_function_id] += 1

            if binary_function_id not in self.binary_func_id_index:
                self.binary_func_id_index.append(binary_function_id)

            try:
                # Load the graph into a JSON object using orjson
                # Read the file in binary mode and parse with orjson
                with open(file_path, "rb") as f:
                    data = orjson.loads(f.read())

                # Iterate over each node within the JSON to remove un-needed data
                for node in data["nodes"]:
                    self.clean_graph_nodes(node)

                # Load JSON into a Rustworkx DiGraph Object
                G = self._load_json_to_rustworkx(data)

                if G.num_edges() == 0:
                    self.no_graphs_failed += 1
                    continue

                # Add edge-betweenness values to each of the edges if enabled (True by default)
                if self.with_edge_features:
                    bb = rx.edge_betweenness_centrality(G, normalized=False)
                    for edge in G.edge_indices():
                        G.update_edge_by_index(edge, {"weight": bb[edge]})

                G = GraphWithMetadata(G)

                # Generate an integer label based on the index of the binary_function_id
                self._generate_graph_label(G, binary_function_id)

                # Add function name to the graph metadata
                G.metadata["name"] = binary_function_id

                # Convert rustworkx graph to a PyTorch Geometric Data object
                group_node_attrs = [
                    "ninstrs",
                    "edges",
                    "indegree",
                    "outdegree",
                    "nlocals",
                    "nargs",
                ]
                pyg = from_rustworkx(
                    G, group_node_attrs=group_node_attrs, group_edge_attrs=["weight"]
                )

                # Validate the graph and append
                try:
                    pyg.validate()
                except:
                    logger.warning(f"Failed to validate {file_path}")

                self.graphs.append(pyg)

            except Exception as e:
                print(f"Failed to load {file_path} - Exception: {e}")
                self.no_graphs_failed += 1

        # Top 10 least common for debugging
        top_10 = sampled_id_counts.most_common()[-10:]
        logger.info(f"Least sampled functions: {top_10}")

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

    def _load_json_to_rustworkx(self, data: dict) -> PyDiGraph:
        """
        Load a JSON graph representation into a Rustworkx DiGraph.

        :param data: The JSON data representing the graph.
        :return: A Rustworkx DiGraph object.
        """
        G = PyDiGraph()
        node_indices = {}

        # Add nodes
        for node in data["nodes"]:
            node_index = G.add_node(node)
            node_indices[node["id"]] = node_index

        # Add edges
        for edge in data["edges"]:
            source = node_indices[edge["source"]]
            target = node_indices[edge["target"]]
            G.add_edge(source, target, None)

        return G

    def get_binary_func_id(self, file_path: str) -> str:
        if self.dataset_naming_convention == "binkit":
            return self.get_binkit_binary_func_id(file_path)
        elif self.dataset_naming_convention == "binarycorp":
            return self.get_binarycorp_binary_func_id(file_path)
        elif self.dataset_naming_convention == "trex":
            return self.get_trex_binary_func_id(file_path)
        elif self.dataset_naming_convention == "custom":
            return self.get_custom_binary_func_id(file_path)
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

    def get_custom_binary_func_id(self, file_path: str) -> str:
        return Path(file_path).name.split("_sub")[0]

    def clean_graph_nodes(self, node):
        if "functionFeatureSubset" not in node or not isinstance(
            node["functionFeatureSubset"], dict
        ):
            return  # Graceful exit if key is missing or not a dict

        function_features = node.pop("functionFeatureSubset", {})
        # Remove specific keys if they exist
        function_features.pop("signature", None)
        function_features.pop("name", None)
        # Merge the remaining items back into the node
        node.update(function_features)

    def save_dataset(self, save_prefix: str):
        with open(f"{save_prefix}-graphs.pickle", "wb") as fp:
            pickle.dump(self.graphs, fp)

        with open(f"{save_prefix}-labels.pickle", "wb") as fp:
            pickle.dump(self.labels, fp)

    def _generate_graph_label(self, G: PyDiGraph, binary_function_id: str):
        """
        Generate the label for a function call graphlet

        :param G: the rustworkx graph to generate metadata for
        :param binary_function_id: the binary name + function id string for the graph
        :return: integer label for the graph
        """
        label = self.binary_func_id_index.index(binary_function_id)
        G.metadata["label"] = label
        self.labels.append(label)
