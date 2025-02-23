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
from statistics import mean
import math


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

        # Add validation data loading
        logger.info("Loading validation data...")
        with open(config.test_data, "rb") as fp:
            self.val_graphs = pickle.load(fp)

        logger.info("Loading validation labels...")
        with open(config.test_labels, "rb") as fp:
            self.val_labels = pickle.load(fp)

        # Create validation DataLoader
        self.val_dataloader = DataLoader(
            self.val_graphs, batch_size=config.batch_size, shuffle=True
        )

        # Early stopping parameters
        self.early_stopping_patience = config.early_stopping_patience
        self.early_stopping_delta = config.early_stopping_delta
        self.epochs_without_improvement = 0
        self.best_val_loss = float("inf")

        # Load model to device
        self.model = model
        model.to(self.device)

    def _validate(self):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for data in self.val_dataloader:
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
                        x=data.x, edge_index=data.edge_index, batch=data.batch
                    )

                # Calculate loss
                hard_pairs = self.miner(embeds, data["label"])
                loss = self.loss_func(embeds, data["label"], hard_pairs)
                total_loss += loss.item()
                num_batches += 1

        avg_val_loss = total_loss / num_batches
        return {"val_loss": avg_val_loss}

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

            # Calculate epoch training metrics
            avg_train_loss = epoch_train_loss / len(self.dataloader)

            # Validation phase
            self.model.eval()
            val_metrics = self._validate()
            avg_val_loss = val_metrics["val_loss"]

            # Update progress bar
            training_progress.set_postfix(
                {
                    "train_loss": f"{avg_train_loss:.4f}",
                    "val_loss": f"{avg_val_loss:.4f}",
                }
            )

            # WandB logging
            if self.log_to_wandb:
                wandb.log(
                    {
                        "epoch/train_loss": avg_train_loss,
                        "epoch/val_loss": avg_val_loss,
                        "epoch": epoch,
                    }
                )

            # Early stopping check
            if (self.best_val_loss - avg_val_loss) > self.early_stopping_delta:
                self.best_val_loss = avg_val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Final save
        self.save_model(prefix="final")
        if self.log_to_wandb:
            wandb.save(f"{self.config.exp_uuid}_*.ep{self.config.epochs}")

    def save_model(self, prefix=""):
        filename = f"models/{self.config.exp_uuid}{f'_{prefix}' if prefix else ''}.ep{self.config.epochs}"
        torch.save(self.model.state_dict(), filename)
