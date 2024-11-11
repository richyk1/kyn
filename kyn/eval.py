import pickle
from collections import Counter
import random

from torchmetrics.retrieval import RetrievalMRR, RetrievalRecall, RetrievalNormalizedDCG

from typing import Union
from kyn.utils import validate_device


from pathlib import Path
from statistics import median, mean
from typing import List, Tuple, Optional
import glob
import json

import torch
import torch.nn as nn
from torch import cosine_similarity
import networkx as nx
from networkx.readwrite import json_graph
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from tqdm import tqdm
from loguru import logger


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
                al = a.label.item()
                a.x = a.x.to(torch.float32)
                a_e = self._gen_embedding(a)

                bl = b.label.item()
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


class KYNVulnEvaluator:
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        target_data_path: str,
        search_data_paths: List[str],
        vulnerable_functions: List[str],
        device: str = "cuda",
        target_arch: str = None,
        no_metadata: bool = False,
        save_metrics_to_file: bool = True,
    ):
        self.model = model
        self.model_name = model_name
        self.target_data_path = target_data_path
        self.search_data_paths = search_data_paths
        self.vulnerable_functions = vulnerable_functions
        self.device = device
        self.target_arch = target_arch
        self.no_metadata = no_metadata
        self.save_metrics_to_file = save_metrics_to_file

        # Initialize empty containers
        self.target_graphs = []
        self.target_embeddings = []
        self.metric_results = []

        self._prepare_model()

    def _prepare_model(self) -> None:
        """Prepare model for evaluation."""
        self.model.eval()
        self.model.to(self.device)
        assert self.model.training is False

    def _process_graph_data(self, data: dict) -> dict:
        """Process graph data by handling metadata."""
        if self.no_metadata:
            for node in data["nodes"]:
                del node["functionFeatureSubset"]
        else:
            for node in data["nodes"]:
                del node["functionFeatureSubset"]["signature"]
                del node["functionFeatureSubset"]["name"]
                for k, v in node["functionFeatureSubset"].items():
                    node[k] = v
                del node["functionFeatureSubset"]
        return data

    def _convert_to_pyg_graph(self, data: dict, file_path: str) -> Optional[Data]:
        """Convert JSON graph data to PyG graph."""
        try:
            G = json_graph.adjacency_graph(data)
            bb = nx.edge_betweenness_centrality(G, normalized=False)
            nx.set_edge_attributes(G, bb, "weight")
            G.graph["name"] = Path(file_path).name.split("-")[0]

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
            pyg.x = pyg.x.to(torch.float32)
            return pyg
        except Exception as e:
            logger.debug(f"Failed to convert graph {file_path}: {str(e)}")
            return None

    def _load_graphs(self, data_path: str, vuln_filter: bool = False) -> List[Data]:
        """Load graphs from JSON files."""
        graphs = []
        graph_paths = glob.glob(f"{data_path}/*.json")

        if vuln_filter:
            graph_paths = [
                p
                for p in graph_paths
                if any(
                    vuln == Path(p).name.split("-")[0][4:]
                    for vuln in self.vulnerable_functions
                )
            ]

        for file_path in tqdm(graph_paths, desc="Loading graphs"):
            try:
                with open(file_path) as fd:
                    data = json.load(fd)
                data = self._process_graph_data(data)
                pyg = self._convert_to_pyg_graph(data, file_path)
                if pyg is not None:
                    graphs.append(pyg)
            except Exception as e:
                logger.debug(f"Failed to load graph {file_path}: {str(e)}")
                continue

        return graphs

    def _generate_embeddings(
        self, graphs: List[Data]
    ) -> List[Tuple[str, torch.Tensor]]:
        """Generate embeddings for a list of graphs."""
        embeddings = []
        with torch.inference_mode():
            for graph in tqdm(graphs, desc="Generating embeddings"):
                graph = graph.to(self.device)
                embed = self.model.forward(
                    x=graph.x,
                    edge_index=graph.edge_index,
                    batch=graph.batch,
                    edge_weight=graph.edge_attr,
                )
                embeddings.append((graph.name, embed))
        return embeddings

    def _compute_rankings(
        self,
        vuln_queries: List[Tuple[str, torch.Tensor]],
        search_embeddings: List[Tuple[str, torch.Tensor]],
    ) -> Tuple[List[int], List[float]]:
        """Compute rankings and similarity scores."""
        ranks = []
        sim_scores = []

        for name, target in vuln_queries:
            similarities = []
            names = []

            for sp_name, sp_embed in search_embeddings:
                sim = cosine_similarity(target, sp_embed)
                similarities.append(sim)
                names.append(sp_name)

            zipped = list(zip(similarities, names))
            zipped.sort(reverse=True)

            for i, (sim, match_name) in enumerate(zipped):
                if name == match_name:
                    ranks.append(i + 1)
                    sim_scores.append(round(sim.item(), 4))
                    logger.info(
                        f"Found {name} at rank {i + 1} with similarity score {round(sim.item(), 4)} to {match_name}"
                    )

        return ranks, sim_scores

    def _write_metrics_to_file(
        self, search_data_path: str, ranks: List[int], sim_scores: List[float]
    ) -> None:
        """Write evaluation metrics to file."""
        if self.save_metrics_to_file:
            compare_arch = search_data_path.split("_")[-2]
            output_path = f"{self.model_name}_{self.target_arch}vs{compare_arch}.txt"

            with open(output_path, "w") as f:
                f.write(
                    f"Ranks: {ranks}\n"
                    f"Mean Rank: {round(mean(ranks))}\n"
                    f"Median Rank: {median(ranks)}\n"
                    f"Similarity Scores: {sim_scores}\n"
                )

    def evaluate(self) -> List[dict]:
        """Run the full evaluation process."""
        # Load and embed target graphs
        logger.info(f"Loading target graphs from {self.target_data_path}")
        self.target_graphs = self._load_graphs(self.target_data_path)
        self.target_embeddings = self._generate_embeddings(self.target_graphs)

        # Evaluate against each search dataset
        for search_data_path in self.search_data_paths:
            logger.info(f"Evaluating against {search_data_path}")

            # Load and embed search pool graphs
            vuln_query_graphs = self._load_graphs(search_data_path, vuln_filter=True)
            vuln_queries = self._generate_embeddings(vuln_query_graphs)

            # Compute rankings and similarities
            ranks, sim_scores = self._compute_rankings(
                vuln_queries, self.target_embeddings
            )

            # Calculate metrics
            metrics = {
                "search_data": search_data_path,
                "ranks": ranks,
                "mean_rank": round(mean(ranks)),
                "median_rank": median(ranks),
                "similarity_scores": sim_scores,
                "mean_similarity": round(mean(sim_scores), 4),
            }

            self.metric_results.append(metrics)

            # Log results
            logger.info(
                f"Results for {search_data_path} Ranks: {metrics['ranks']} "
                f"Mean Rank: {metrics['mean_rank']}\n"
                f"Median Rank: {metrics['median_rank']}\n"
                f"Mean Similarity: {metrics['mean_similarity']}"
            )

            # Write metrics to file if requested
            self._write_metrics_to_file(search_data_path, ranks, sim_scores)

        return self.metric_results


if __name__ == "__main__":
    from kyn.networks import GraphConvInstanceGlobalMaxSmallSoftMaxAggrEdge
    import os

    # Example usage
    MODEL_PATH = "../cc4c0267.ep20"
    model = GraphConvInstanceGlobalMaxSmallSoftMaxAggrEdge(256, 6)
    model_dict = torch.load(MODEL_PATH)
    model.load_state_dict(model_dict)

    TPLINK_VULNS = [
        "CMS_decrypt",
        "PKCS7_dataDecode",
        "BN_bn2dec",
        "EVP_EncodeUpdate",
        "BN_dec2bn",
        "BN_hex2bn",
    ]

    NETGEAR_VULNS = ["CMS_decrypt", "PKCS7_dataDecode", "MDC2_Update", "BN_bn2dec"]

    # Define paths
    DATA_ROOT = "/fast-disk/Dataset-Vuln/cgs"
    TPLINK_DATA = os.path.join(
        DATA_ROOT,
        "libcrypto.so.1.0.0_TP-Link_Deco-M4_1.0.2d_mips32_cg-onehopcgcallers-meta",
    )
    TARGET_DATA = os.path.join(
        DATA_ROOT,
        "libcrypto.so.1.0.0_NETGEAR_R7000_1.0.2h_arm32_cg-onehopcgcallers-meta",
    )
    SEARCH_PATHS = [
        os.path.join(
            DATA_ROOT,
            "libcrypto.so.1.0.0_openssl_1.0.2d_mips32_cg-onehopcgcallers-meta",
        ),
        os.path.join(
            DATA_ROOT, "libcrypto.so.1.0.0_openssl_1.0.2d_x64_cg-onehopcgcallers-meta"
        ),
        os.path.join(
            DATA_ROOT, "libcrypto.so.1.0.0_openssl_1.0.2d_x86_cg-onehopcgcallers-meta"
        ),
        os.path.join(
            DATA_ROOT, "libcrypto.so.1.0.0_openssl_1.0.2d_arm32_cg-onehopcgcallers-meta"
        ),
        os.path.join(
            DATA_ROOT, "libcrypto.so.1.0.0_openssl_1.0.2d_ppc32_cg-onehopcgcallers-meta"
        ),
        os.path.join(
            DATA_ROOT,
            "libcrypto.so.1.0.0_openssl_1.0.0d_riscv32_cg-onehopcgcallers-meta",
        ),
        # Add other paths as needed
    ]

    # Create evaluator instance
    evaluator = KYNVulnEvaluator(
        model=model,
        model_name="best-cross",
        target_data_path=TPLINK_DATA,
        search_data_paths=SEARCH_PATHS,
        vulnerable_functions=TPLINK_VULNS,
        target_arch="arm32",
    )

    # Run evaluation
    results = evaluator.evaluate()
