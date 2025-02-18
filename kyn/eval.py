import glob
import json
import os
import pickle
import random
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import List, Optional, Tuple, Union

import orjson
import rustworkx as rx
import torch
import torch.nn as nn
from from_rustworkx import from_rustworkx
from GraphWithMetadata import GraphWithMetadata
from kyn.networks import GraphConvInstanceGlobalMaxSmallSoftMaxAggrEdge
from kyn.utils import validate_device
from loguru import logger
from networkx.readwrite import json_graph
from torch import cosine_similarity
from torch_geometric.data import Data
from torchmetrics.retrieval import RetrievalMRR, RetrievalNormalizedDCG, RetrievalRecall
from tqdm import tqdm


class KYNEvaluator:
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        dataset_path: str,
        eval_prefix: str,
        device: str = None,
        search_pool_size: Union[int, list[int]] = 10,
        num_search_pools: int = 1,
        random_seed: int = 1337,
        requires_edge_feats: bool = True,
        save_metrics_to_file: bool = True,
    ):
        self.model = model
        self.device = validate_device(device)
        self._prepare_model()
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.experiment_prefix = eval_prefix
        self.search_pool_size = search_pool_size
        self.num_search_pools = num_search_pools
        self.random_seed = random_seed
        self.requires_edge_feats = requires_edge_feats
        self.save_metrics_to_file = save_metrics_to_file

        self._load_eval_graphs_and_labels()

        self._preprocess_data()

        self.indexes = None
        self.targets = None
        self.scores = None
        self.metric_dicts = []

        random.seed(self.random_seed)

    def _prepare_model(self) -> None:
        self.model.eval()
        self.model.to(self.device)
        assert self.model.training is False

    def _load_eval_graphs_and_labels(self):
        with open(f"{self.dataset_path}-graphs.pickle", "rb") as fp:
            self.graphs = pickle.load(fp)

        with open(f"{self.dataset_path}-labels.pickle", "rb") as fp:
            self.labels = pickle.load(fp)

    def _preprocess_data(self):
        # Sort graphs and labels
        self.graphs = list(sorted(self.graphs, key=lambda x: x.label))
        self.labels = list(sorted(self.labels))

        self.num_graphs = len(self.graphs)
        self.label_counter = Counter(self.labels)

        # Get cases where we only have one graph of a particular label
        self.no_singletons_labels = [k for k, v in self.label_counter.items() if v >= 2]

    def _sample_search_pool(self, label: int, search_pool_size: int) -> list[Data]:
        pos_pair = []
        neg_pairs = []
        offset_samples = random.sample(range(self.label_counter[label]), k=2)

        a = self.graphs[self.labels.index(label) + offset_samples[0]]["label"]

        b = self.graphs[self.labels.index(label) + offset_samples[1]]["label"]
        assert a == b

        pos_pair.append(
            (
                self.graphs[self.labels.index(label) + offset_samples[0]],
                self.graphs[self.labels.index(label) + offset_samples[1]],
            )
        )

        while len(neg_pairs) != search_pool_size:
            random_index = random.sample(range(self.num_graphs), k=1)[0]
            if (
                self.graphs[self.labels.index(label) + offset_samples[0]]["label"]
                != self.graphs[random_index]["label"]
            ):
                neg_pairs.append(
                    (
                        self.graphs[self.labels.index(label) + offset_samples[0]],
                        self.graphs[random_index],
                    )
                )
            else:
                continue

        return pos_pair + neg_pairs

    def _generate_single_search_pool_metrics(
        self,
        search_pool: list,
        index: int,
    ) -> None:
        for a, b in search_pool:
            with torch.inference_mode():
                a.to(self.device)
                b.to(self.device)
                al = a.label
                a.x = a.x.to(torch.float32)
                a_e = self._gen_embedding(a)

                bl = b.label
                b.x = b.x.to(torch.float32)
                b_e = self._gen_embedding(b)

                sim = torch.cosine_similarity(a_e, b_e, dim=1).item()
                self.scores.append(sim)
                self.targets.append(True if bl == al else False)

                self.indexes.append(index)

    def _gen_embedding(self, data: Data) -> torch.Tensor:
        if self.requires_edge_feats:
            return self.model.forward(
                x=data.x,
                edge_index=data.edge_index,
                batch=data.batch,
                edge_weight=data.edge_attr,
            )
        else:
            return self.model.forward(
                x=data.x, edge_index=data.edge_index, batch=data.batch
            )

    def _check_previous_stats_presence(self) -> None:
        if (
            self.scores is not None
            or self.targets is not None
            or self.indexes is not None
        ):
            logger.debug(
                "Previous scores, targets and indexes found. Overwriting existing scores..."
            )

        self.scores = []
        self.targets = []
        self.indexes = []

    def _write_metrics_to_file(self, metric_dict: dict, search_pool_size: int) -> None:
        if Path("results/").exists() is False:
            Path("results/").mkdir(parents=True, exist_ok=True)

        try:
            with open(
                f"results/{self.model_name}-{self.experiment_prefix}-{search_pool_size}.log",
                "w",
            ) as f:
                json.dump(metric_dict, f)
        except Exception as e:
            print(e)

    def _generate_metrics(self, search_pool_size: int):
        indexes, scores, target = (
            torch.LongTensor(self.indexes),
            torch.FloatTensor(self.scores),
            torch.LongTensor(self.targets),
        )

        metric_dict = {}

        metric_name_func_tuples = [
            (
                f"{self.experiment_prefix}-{search_pool_size}-MRR@10",
                RetrievalMRR(top_k=10),
            ),
            (
                f"{self.experiment_prefix}-{search_pool_size}-R@1",
                RetrievalRecall(top_k=1),
            ),
            (
                f"{self.experiment_prefix}-{search_pool_size}-NDCG@10",
                RetrievalNormalizedDCG(top_k=10),
            ),
        ]

        for name, func in metric_name_func_tuples:
            metric_dict[name] = func(scores, target, indexes=indexes).item()

        metric_dict["highest_sim"] = max(scores).item()
        metric_dict["lowest_sim"] = min(scores).item()

        if self.save_metrics_to_file:
            self._write_metrics_to_file(metric_dict, search_pool_size)

        return metric_dict

    def evaluate(self):
        if isinstance(self.search_pool_size, int):
            self.search_pool_size = [self.search_pool_size]

        for search_pool_size in self.search_pool_size:
            self._check_previous_stats_presence()

            pos_labels = random.choices(
                self.no_singletons_labels, k=self.num_search_pools
            )

            for i, label in tqdm(enumerate(pos_labels), total=self.num_search_pools):
                search_pool = self._sample_search_pool(label, search_pool_size)
                self._generate_single_search_pool_metrics(search_pool, i)

            metrics = self._generate_metrics(search_pool_size)
            self.metric_dicts.append((search_pool_size, metrics))

        for metric_dict in self.metric_dicts:
            logger.info(f"Metrics for ({metric_dict[0]}): {metric_dict[1]}")


class CosineSimilarityEvaluator:
    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self._prepare_model()

    def _prepare_model(self) -> None:
        """Prepare model for evaluation."""
        self.model.eval()
        self.model.to(self.device)
        assert self.model.training is False

    def _load_json_to_rustworkx(self, data: dict) -> rx.PyDiGraph:
        """
        Load a JSON graph representation into a Rustworkx DiGraph.
        """
        G = rx.PyDiGraph()
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

    def _generate_embeddings(self, graph_data) -> torch.Tensor:
        """
        Generate embeddings for a single graph.
        Returns a tensor of shape (1, embedding_dim).
        """
        with torch.inference_mode():
            graph_data = graph_data.to(self.device)

            embed = self.model(
                x=graph_data.x,
                edge_index=graph_data.edge_index,
                batch=graph_data.batch,
                edge_weight=graph_data.edge_attr,
            )
            return embed

    def _load_and_transform_graph(self, file_path: str) -> torch.Tensor:
        """
        Load a JSON graph from 'file_path', transform to PyG data, and generate its embedding.
        Returns the embedding tensor of shape (1, embedding_dim).
        """
        try:
            # Load JSON graph
            with open(file_path, "rb") as f:
                data = orjson.loads(f.read())

            # Convert JSON -> rustworkx
            G = self._load_json_to_rustworkx(data)

            # (Optional) compute edge betweenness or other features
            bb = rx.edge_betweenness_centrality(G, normalized=False)
            for edge_idx in G.edge_indices():
                G.update_edge_by_index(edge_idx, {"weight": bb[edge_idx]})

            # Wrap in our custom GraphWithMetadata, if needed
            G = GraphWithMetadata(G)
            G.metadata["name"] = os.path.basename(file_path)

            # Convert rustworkx -> PyTorch Geometric Data
            group_node_attrs = [
                "ninstrs",
                "edges",
                "indegree",
                "outdegree",
                "nlocals",
                "nargs",
            ]
            pyg_data = from_rustworkx(
                G, group_node_attrs=group_node_attrs, group_edge_attrs=["weight"]
            )
            pyg_data.validate()

            # Generate the embedding
            embedding = self._generate_embeddings(pyg_data)
            return embedding

        except Exception as e:
            logger.error(f"Failed to load or transform {file_path}: {e}")
            return None

    def compute_similarity(
        self, embedding1: torch.Tensor, embedding2: torch.Tensor, dim: int = 1
    ) -> float:
        """
        Compute the cosine similarity between two embeddings.
        """
        # Ensure the embeddings are of floating point type
        embedding1 = embedding1.float()
        embedding2 = embedding2.float()

        # Compute cosine similarity
        similarity = torch.cosine_similarity(embedding1, embedding2, dim=dim)
        if similarity.numel() == 1:
            return similarity.item()
        return similarity

    def compare_one_to_many(
        self,
        target_file: str,
        search_dir: str,
        top_k: int = 5,
        max_files: int = -1,  # <--- -1 means "no limit"
    ):
        """
        1. Load and embed the 'target_file'.
        2. For each file in 'search_dir', load and embed, then compute similarity
           to the target embedding.
        3. Sort results by similarity (descending) and return top_k.

        :param target_file: Path to the single target JSON file.
        :param search_dir: Directory containing candidate JSON files to compare against.
        :param top_k: How many of the most similar functions to return.
        :param max_files: Maximum number of files to process. -1 means no limit.
        :param valid_extensions: Tuple of valid file extensions.
        :return: A list of tuples (similarity, filename) for the top_k similar graphs.
        """
        # 1) Get target embedding
        target_embedding = self._load_and_transform_graph(target_file)
        if target_embedding is None:
            logger.error(f"Failed to produce embedding for target {target_file}")
            return []

        # 2) Collect similarities
        similarity_scores = []
        search_dir_path = Path(search_dir)
        if not search_dir_path.exists() or not search_dir_path.is_dir():
            logger.error(
                f"Search directory {search_dir} does not exist or is not a directory."
            )
            return []

        count = 0
        file_list = sorted(search_dir_path.glob("*"))

        for file_path in tqdm(file_list, desc="Comparing graphs"):
            # Stop early if we've reached the max_files limit
            if max_files != -1 and count >= max_files:
                logger.debug(f"Stopping early (max_files={max_files})")
                break

            # Generate embedding for this search graph
            candidate_embedding = self._load_and_transform_graph(str(file_path))
            if candidate_embedding is None:
                # Skip failures
                logger.warning(f"Failure during canditate embedding creation")
                continue

            # Compute similarity
            sim_score = self.compute_similarity(target_embedding, candidate_embedding)
            similarity_scores.append((sim_score, file_path.name))

            count += 1

        # 3) Sort by descending similarity
        similarity_scores.sort(key=lambda x: x[0], reverse=True)

        # Find the rank of the actual target function
        target_filename = os.path.basename(target_file)
        actual_rank = next(
            (
                i + 1
                for i, (_, fname) in enumerate(similarity_scores)
                if fname == target_filename
            ),
            None,
        )

        # Print the rank of the actual function
        if actual_rank:
            print(
                f"Actual function '{target_filename}' is ranked at position {actual_rank} out of {len(similarity_scores)}."
            )
        else:
            print(f"Actual function '{target_filename}' not found in the ranking.")

        # Return the top_k most similar functions (same as before)
        return similarity_scores[:top_k]


if __name__ == "__main__":
    import torch
    from kyn.networks import GraphConvInstanceGlobalMaxSmallSoftMaxAggrEdge

    MODEL_PATH = "./1e7b2a8d.ep350"
    model = GraphConvInstanceGlobalMaxSmallSoftMaxAggrEdge(256, 6)
    model.load_state_dict(torch.load(MODEL_PATH))

    evaluator = CosineSimilarityEvaluator(model=model, device="cuda")

    target_file = (
        "/home/prime/dev/edges/cgn/eu4_arm_1.27.2/CPdxMouse::LoadCursors_subgraph.json"
    )
    search_folder = "/home/prime/dev/edges/cgn/eu4_arm_1.35.2/"

    top_k_results = evaluator.compare_one_to_many(
        target_file=target_file, search_dir=search_folder, top_k=5, max_files=-1
    )

    print("Top 5 Most Similar Functions:")
    for score, fname in top_k_results:
        print(f"{fname} -> similarity={score:.4f}")
