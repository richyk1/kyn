import json
import pickle
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Union

import orjson
import torch
import torch.nn as nn
from kyn.config import KYNConfig
from kyn.networks import GraphConvInstanceGlobalMaxSmallSoftMaxAggrEdge
from kyn.utils import validate_device
from loguru import logger
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
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
        self._precompute_embeddings()

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

        self.no_multi_labels = [k for k, v in self.label_counter.items() if v >= 2]

        self.label_to_indices = defaultdict(list)
        for idx, lbl in enumerate(self.labels):
            self.label_to_indices[lbl].append(idx)
        # Assign index to graphs
        for idx, g in enumerate(self.graphs):
            g.index = idx

    def _precompute_embeddings(self):
        self.model.eval()
        all_embeddings = []
        batch_size = KYNConfig().batch_size
        data_loader = DataLoader(self.graphs, batch_size=batch_size, shuffle=False)
        for batch in tqdm(data_loader):
            batch = batch.to(self.device)
            with torch.no_grad():
                embs = self.model(
                    x=batch.x.float(),  # Ensure correct dtype
                    edge_index=batch.edge_index,
                    edge_weight=batch.edge_attr if self.requires_edge_feats else None,
                    batch=batch.batch,
                )
            all_embeddings.append(embs.cpu())
        self.embeddings = torch.cat(all_embeddings).to(self.device)

    def _sample_search_pool(self, label: int, search_pool_size: int) -> list[Data]:
        current_indices = self.label_to_indices[label]
        a_idx, b_idx = random.sample(current_indices, 2)
        a, b = self.graphs[a_idx], self.graphs[b_idx]
        pos_pairs = [(a, b)]

        # Get all indices excluding current label
        all_indices = set(range(self.num_graphs))
        negative_indices = list(all_indices - set(current_indices))
        sampled_neg_indices = random.sample(negative_indices, search_pool_size)
        neg_pairs = [(a, self.graphs[i]) for i in sampled_neg_indices]

        return pos_pairs + neg_pairs

    def _generate_single_search_pool_metrics(
        self, search_pool: list, index: int
    ) -> None:
        a_indices = [a.index for a, _ in search_pool]
        b_indices = [b.index for _, b in search_pool]

        a_embs = self.embeddings[a_indices]
        b_embs = self.embeddings[b_indices]

        sims = torch.cosine_similarity(a_embs, b_embs, dim=1).tolist()
        targets = [
            self.labels[a] == self.labels[b] for a, b in zip(a_indices, b_indices)
        ]

        self.scores.extend(sims)
        self.targets.extend(targets)
        self.indexes.extend([index] * len(sims))

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

        datetime_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        try:
            with open(
                f"results/{self.model_name}-{self.experiment_prefix}-{search_pool_size}-{datetime_str}.json",
                "w",
            ) as f:
                json.dump(metric_dict, f, indent=2)
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

            pos_labels = random.choices(self.no_multi_labels, k=self.num_search_pools)

            for i, label in tqdm(enumerate(pos_labels), total=self.num_search_pools):
                search_pool = self._sample_search_pool(label, search_pool_size)
                self._generate_single_search_pool_metrics(search_pool, i)

            metrics = self._generate_metrics(search_pool_size)
            self.metric_dicts.append((search_pool_size, metrics))

        for metric_dict in self.metric_dicts:
            logger.info(f"Metrics for ({metric_dict[0]}): {metric_dict[1]}")


def get_model(model_name: str, config: KYNConfig) -> torch.nn.Module:
    """Get the appropriate model based on the model name."""
    models = {
        "GraphConvInstanceGlobalMaxSmallSoftMaxAggrEdge": GraphConvInstanceGlobalMaxSmallSoftMaxAggrEdge,
        # "GraphConvGraphNormGlobalMaxSmallSoftMaxAggrEdge": GraphConvGraphNormGlobalMaxSmallSoftMaxAggrEdge,
        # "GraphConvLayerNormGlobalMaxSmallSoftMaxAggrEdge": GraphConvLayerNormGlobalMaxSmallSoftMaxAggrEdge,
    }

    if model_name not in models:
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {list(models.keys())}"
        )

    return models[model_name](
        config.model_channels, config.feature_dim, config.dropout_ratio
    )


if __name__ == "__main__":
    """Generate and save a dataset."""

    # python cli.py evaluate --model-path models/eb7f7a5f_sweep.ep300 --model-name GraphConvInstanceGlobalMaxSmallSoftMaxAggrEdge --dataset-path datasets/mixed_validation_30_000 --eval-prefix eu4_140_000 --requires-edge-feats

    """Evaluate a trained model."""
    model = get_model(
        "GraphConvInstanceGlobalMaxSmallSoftMaxAggrEdge",
        KYNConfig(),
    )
    model.load_state_dict(torch.load("models/eb7f7a5f_sweep.ep300"))

    evaluator = KYNEvaluator(
        model=model,
        model_name="GraphConvInstanceGlobalMaxSmallSoftMaxAggrEdge",
        dataset_path="datasets/mixed_validation_30_000",
        eval_prefix="win_mac",
        search_pool_size=[100, 1000, 10000],
        num_search_pools=1000,
        random_seed=1337,
        requires_edge_feats=True,
    )

    evaluator.evaluate()
