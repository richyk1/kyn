import os
import torch
from pathlib import Path

import orjson
import rustworkx as rx
import torch
from from_rustworkx import from_rustworkx
from GraphWithMetadata import GraphWithMetadata
from kyn.config import KYNConfig
from kyn.networks import GraphConvInstanceGlobalMaxSmallSoftMaxAggrEdge
from loguru import logger
from tqdm import tqdm
from torch_geometric.data import Batch
from kyn.networks import GraphConvInstanceGlobalMaxSmallSoftMaxAggrEdge


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

            return pyg_data

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
        max_files: int = -1,
    ):
        # 1) Load PyG Data for the target (NOT the embedding yet).
        target_pyg_data = self._load_and_transform_graph(target_file)
        if target_pyg_data is None:
            logger.error(f"Failed to create PyG Data for target {target_file}")
            return []

        # 2) Collect PyG Data for each candidate
        search_dir_path = Path(search_dir)
        if not search_dir_path.exists() or not search_dir_path.is_dir():
            logger.error(
                f"Search directory {search_dir} does not exist or is not a directory."
            )
            return []

        file_list = sorted(search_dir_path.glob("*"))

        all_data_list = []
        all_filenames = []

        count = 0
        for file_path in tqdm(file_list, desc="Collecting PyG data"):
            if max_files != -1 and count >= max_files:
                break

            candidate_pyg_data = self._load_and_transform_graph(str(file_path))
            if candidate_pyg_data is None:
                continue

            all_data_list.append(candidate_pyg_data)
            all_filenames.append(file_path.name)
            count += 1

        if not all_data_list:
            logger.warning("No valid candidate graphs found.")
            return []

        # 3) Convert the *target* into a single-item Batch and get its embedding
        target_batch = Batch.from_data_list([target_pyg_data]).to(self.device)
        with torch.inference_mode():
            target_embedding = self.model(
                x=target_batch.x,
                edge_index=target_batch.edge_index,
                batch=target_batch.batch,
                edge_weight=target_batch.edge_attr,
            )
        # target_embedding shape -> (1, D)

        # 4) Batch all candidate graphs in one forward pass
        candidate_batch = Batch.from_data_list(all_data_list).to(self.device)

        with torch.inference_mode():
            candidate_embeddings = self.model(
                x=candidate_batch.x,
                edge_index=candidate_batch.edge_index,
                batch=candidate_batch.batch,
                edge_weight=candidate_batch.edge_attr,
            )
        # candidate_embeddings shape -> (N, D)

        # 5) Compute similarity in a vectorized manner
        # target_embedding is (1, D), candidate_embeddings is (N, D)
        # -> sim_scores shape (N,)
        sim_scores = torch.cosine_similarity(
            target_embedding, candidate_embeddings, dim=1
        )
        sim_scores = sim_scores.cpu().tolist()

        similarity_scores = list(zip(sim_scores, all_filenames))
        similarity_scores.sort(key=lambda x: x[0], reverse=True)

        return similarity_scores[:top_k]


def get_model(model_name: str, config: KYNConfig) -> torch.nn.Module:
    """Get the appropriate model based on the model name."""
    models = {
        "GraphConvInstanceGlobalMaxSmallSoftMaxAggrEdge": GraphConvInstanceGlobalMaxSmallSoftMaxAggrEdge,
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

    MODEL_PATH = "./models/baf95786.ep1200"
    model = GraphConvInstanceGlobalMaxSmallSoftMaxAggrEdge(256, 6)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

    evaluator = CosineSimilarityEvaluator(model=model, device="cuda")

    target_file = "/home/prime/dev/edges/test_cgn/eu4_mac_1.37.2/CCountry::SetArmyTradition_subgraph.json"
    search_folder = "/home/prime/dev/edges/test_cgn/eu4_win_1.37.2/"

    top_k_results = evaluator.compare_one_to_many(
        target_file=target_file, search_dir=search_folder, top_k=5, max_files=-1
    )

    print("Top 5 Most Similar Functions:")
    for score, fname in top_k_results:
        print(f"{fname} -> similarity={score:.4f}")
