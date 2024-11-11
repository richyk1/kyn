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

from kyn.networks import GraphConvInstanceGlobalMaxSmallSoftMaxAggrEdge


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

        self.model = model
        model.to(self.device)

    def train(self, validate_examples: bool = False):
        losses = []
        batch_losses = []

        for epoch in tqdm(range(self.config.epochs)):
            self.optim.zero_grad()
            if len(losses) > 0:
                if self.log_to_wandb:
                    wandb.log(
                        {"avg_loss": mean(losses), "batch_loss": mean(batch_losses)}
                    )
                batch_losses = []

            for i, data in enumerate(self.dataloader):
                if validate_examples:
                    data.validate()

                data.to("cuda")

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

                hard_pairs = self.miner(embeds, data["label"])

                loss = self.loss_func(embeds, data["label"], hard_pairs)

                if self.log_to_wandb:
                    wandb.log({"loss": loss.item()})

                losses.append(loss.item())
                batch_losses.append(loss.item())
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()
                self.scheduler.step(epoch + i / self.num_iters)

                if self.log_to_wandb:
                    wandb.log({"current_lr": self.optim.param_groups[0]["lr"]})

    def save_model(self):
        torch.save(
            self.model.state_dict(), f"{self.config.exp_uuid}.ep{self.config.epochs}"
        )


if __name__ == "__main__":
    config = KYNConfig(
        train_data="../datasets/dummy/binkit-test-new-class-small-graphs.pickle",
        train_labels="../datasets/dummy/binkit-test-new-class-small-labels.pickle",
    )
    model = GraphConvInstanceGlobalMaxSmallSoftMaxAggrEdge(
        config.model_channels, in_channels=6
    )
    trainer = KYNTrainer(model, config, "cuda")
    trainer.train()
