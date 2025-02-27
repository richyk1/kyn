import torch.nn as nn
import wandb
from loguru import logger
import pickle
from kyn.config import KYNConfig
from pytorch_metric_learning.losses import CircleLoss
from pytorch_metric_learning.miners import BatchHardMiner
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.samplers import MPerClassSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.loader import DataLoader
from kyn.utils import validate_device
import torch
from tqdm import tqdm
import faiss
import numpy as np


class KYNTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: KYNConfig,
        device: str,
        log_to_wandb: bool = False,
        wandb_project: str = None,
    ):
        self.config = config
        self.device = validate_device(device)
        self.log_to_wandb = log_to_wandb

        if self.log_to_wandb is True and wandb_project is not None:
            logger.info("Setting up wandb logger...")
            wandb.init(
                # set the wandb project where this run will be logged
                project=wandb_project,
                # track hyperparameters and run metadata
                config=self.config,
            )

        logger.info("Loading up train Data objects...")
        with open(config.train_data, "rb") as fp:
            self.graphs = pickle.load(fp)

        logger.info("Loading up train labels...")
        with open(config.train_labels, "rb") as fp:
            self.labels = pickle.load(fp)

        self.loss_func = CircleLoss(
            m=config.circle_loss_m, gamma=config.circle_loss_gamma
        )
        self.miner = BatchHardMiner(distance=CosineSimilarity())
        self.optim = Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=1e-5
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optim, 50, 2, eta_min=config.min_learning_rate
        )

        if config.sampler_epoch_size == 0:
            config.sampler_epoch_size = len(self.labels)

        self.sampler = MPerClassSampler(
            self.labels,
            config.num_examples_in_batch,
            config.batch_size,
            length_before_new_iter=config.sampler_epoch_size,
        )
        self.dataloader = DataLoader(
            self.graphs, batch_size=config.batch_size, sampler=self.sampler
        )
        self.num_iters = len(self.dataloader)

        if config.test_data and config.test_labels:
            # Add validation data loading
            logger.info("Loading validation data...")
            with open(config.test_data, "rb") as fp:
                self.val_graphs = pickle.load(fp)

            logger.info("Loading validation labels...")
            with open(config.test_labels, "rb") as fp:
                self.val_labels = pickle.load(fp)

            # Create validation DataLoader
            self.val_dataloader = DataLoader(
                self.val_graphs, batch_size=config.batch_size, shuffle=False
            )

            # Early stopping parameters
            self.early_stopping_patience = config.early_stopping_patience
            self.early_stopping_delta = config.early_stopping_delta
            self.epochs_without_improvement = 0
            self.best_recall = 0.0

        self.faiss_res = None
        if self.device == "cuda":
            self.faiss_res = faiss.StandardGpuResources()

        # Load model to device
        self.model = model
        model.to(self.device)

    def _validate(self):
        # Split into query and index sets (adjust ratio as needed)
        split_ratio = 0.5
        split_idx = int(len(self.val_graphs) * split_ratio)
        query_graphs = self.val_graphs[:split_idx]
        index_graphs = self.val_graphs[split_idx:]

        # Generate embeddings for both sets
        query_embeds, query_labels = self._get_embeddings(query_graphs)
        index_embeds, index_labels = self._get_embeddings(index_graphs)

        # --- Calculate Validation Loss ---
        # Compare query embeddings to index embeddings with the same label
        val_losses = []
        self.model.eval()
        with torch.no_grad():
            # Compute distances between matching pairs
            for query_embed, query_label in zip(query_embeds, query_labels):
                # Find all index embeddings with the same label
                same_label_idx = index_labels == query_label
                if np.any(same_label_idx):
                    positive_dists = np.linalg.norm(
                        index_embeds[same_label_idx] - query_embed, axis=1
                    )
                    val_loss = np.mean(
                        positive_dists
                    )  # Loss ~ mean distance to positives
                    val_losses.append(val_loss)

        avg_val_loss = (
            np.mean(val_losses) if val_losses else 0.0
        )  # Default to 0 if no positives

        # Build FAISS index and compute recall@1 (unchanged)
        index = faiss.IndexFlatIP(index_embeds.shape[1])
        faiss.normalize_L2(index_embeds)  # Normalize before adding
        index.add(index_embeds)
        D, I = index.search(query_embeds, 1)  # Top-1 match from index set
        matches = index_labels[I.flatten()]
        recall_at_1 = (matches == query_labels).mean()

        return {
            "val_loss": avg_val_loss,  # Now a numeric value (float)
            "recall_at_1": recall_at_1,
        }

    def _get_embeddings(self, graphs):
        loader = DataLoader(graphs, batch_size=self.config.batch_size)
        all_embeds, all_labels = [], []
        self.model.eval()
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                if self.config.with_edges:
                    embeds = self.model(
                        x=data.x,
                        edge_index=data.edge_index,
                        batch=data.batch,
                        edge_weight=data.edge_attr,
                    )
                else:
                    embeds = self.model(
                        x=data.x,
                        edge_index=data.edge_index,
                        batch=data.batch,
                    )
                all_embeds.append(embeds.cpu().numpy())
                all_labels.append(data["label"].cpu().numpy())
        return np.concatenate(all_embeds), np.concatenate(all_labels)

    def train(self, validate_examples: bool = False):
        training_progress = tqdm(range(self.config.epochs), desc="Training Progress")

        for epoch in training_progress:
            # Training phase
            self.model.train()
            epoch_train_loss = 0.0
            batch_progress = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch+1}/{self.config.epochs}",
                leave=False,
            )

            for i, data in enumerate(batch_progress):
                self.optim.zero_grad()

                if validate_examples:
                    data.validate()

                # Move data to device
                data = data.to(self.device)

                # Forward pass
                if self.config.with_edges:
                    embeds = self.model(
                        x=data.x,
                        edge_index=data.edge_index,
                        batch=data.batch,
                        edge_weight=data.edge_attr,
                    )
                else:
                    embeds = self.model(
                        x=data.x, edge_index=data.edge_index, batch=data.batch
                    )

                # Calculate loss
                hard_pairs = self.miner(embeds, data["label"])
                loss = self.loss_func(embeds, data["label"], hard_pairs)

                # Backward pass and optimize
                loss.backward()
                self.optim.step()
                self.scheduler.step(epoch + i / self.num_iters)

                # Update metrics
                epoch_train_loss += loss.item()
                current_lr = self.optim.param_groups[0]["lr"]

                # Update batch progress description
                batch_progress.set_postfix(
                    {"batch_loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"}
                )

                # Log batch metrics to wandb
                if self.log_to_wandb:
                    wandb.log(
                        {
                            "batch/train_loss": loss.item(),
                            "batch/learning_rate": current_lr,
                        }
                    )

            if self.config.test_data and self.config.test_labels:
                avg_train_loss = epoch_train_loss / len(self.dataloader)
                val_metrics = self._validate()
                avg_val_loss = val_metrics["val_loss"]
                recall_at_1 = val_metrics["recall_at_1"]

                training_progress.set_postfix(
                    {
                        "train_loss": f"{avg_train_loss:.4f}",
                        "val_loss": f"{avg_val_loss:.4f}",
                        "R@1": f"{recall_at_1:.4f}",
                    }
                )

                if self.log_to_wandb:
                    wandb.log(
                        {
                            "epoch/train_loss": avg_train_loss,
                            "epoch/val_loss": avg_val_loss,
                            "epoch/recall_at_1": recall_at_1,
                            "epoch": epoch,
                        }
                    )

                # Instead of val_loss, use recall@1 for early stopping
                if (recall_at_1 - self.best_recall) > self.early_stopping_delta:
                    self.best_recall = recall_at_1
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1

                if self.epochs_without_improvement >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break

    def save_model(self, prefix=""):
        filename = f"models/{self.config.exp_uuid}{f'_{prefix}' if prefix else ''}.ep{self.config.epochs}"
        torch.save(self.model.state_dict(), filename)
