from dataclasses import dataclass, asdict
from typing import Dict, Any
import uuid


@dataclass
class KYNConfig:
    learning_rate: float = 1e-4
    min_learning_rate: float = 2e-6
    model_arch: str = "GraphConvInstanceGlobalMaxSmallVariancePreservingEdge"
    data_type: str = "onehopwithcallers"
    train_data: str = "datasets/dummy/binkit-test-new-class-small-graphs.pickle"
    train_labels: str = "datasets/dummy/binkit-test-new-class-small-labels.pickle"
    test_data: str = ""
    test_labels: str = ""
    epochs: int = 300
    loss: str = "Circle"
    miner: str = "BatchHard"
    optim: str = "Adam"
    batch_size: int = 256
    model_channels: int = 512
    pooling: str = "max"
    circle_loss_m: float = 0.40
    circle_loss_gamma: int = 256
    num_examples_in_batch: int = 4  # full datasaet = 9, 140_000 = 4
    exp_uuid: str = str(uuid.uuid4())[:8]
    feature_dim: int = 6
    dropout_ratio: float = 0.3
    number_eval_sp: int = 500
    with_edges: bool = True
    sampler_epoch_size: int = 0
    early_stopping_patience: int = (
        10  # Number of epochs to wait, set to 50 on big build
    )
    early_stopping_delta: float = 0.01  # Minimum improvement threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert the dataclass instance to a dictionary.
        Filters out None values by default."""
        return {k: v for k, v in asdict(self).items() if v is not None}
