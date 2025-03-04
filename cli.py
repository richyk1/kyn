import argparse
from pathlib import Path

import torch
from loguru import logger
import wandb
import yaml

from kyn.dataset import KYNDataset
from kyn.trainer import KYNTrainer
from kyn.networks import (
    GraphConvInstanceGlobalMaxSmall,
    GraphConvInstanceGlobalMaxSmallSoftMaxAggr,
    GraphConvInstanceGlobalMaxSmallSoftMaxAggrEdge,
    GraphConvGraphNormGlobalMaxSmallSoftMaxAggrEdge,
    GraphConvLayerNormGlobalMaxSmallSoftMaxAggrEdge,
)
from kyn.config import KYNConfig
from kyn.eval import KYNEvaluator


def get_model(model_name: str, config: KYNConfig) -> torch.nn.Module:
    """Get the appropriate model based on the model name."""
    models = {
        "GraphConvInstanceGlobalMaxSmall": GraphConvInstanceGlobalMaxSmall,
        "GraphConvInstanceGlobalMaxSmallSoftMaxAggr": GraphConvInstanceGlobalMaxSmallSoftMaxAggr,
        "GraphConvInstanceGlobalMaxSmallSoftMaxAggrEdge": GraphConvInstanceGlobalMaxSmallSoftMaxAggrEdge,
        "GraphConvGraphNormGlobalMaxSmallSoftMaxAggrEdge": GraphConvGraphNormGlobalMaxSmallSoftMaxAggrEdge,
        "GraphConvLayerNormGlobalMaxSmallSoftMaxAggrEdge": GraphConvLayerNormGlobalMaxSmallSoftMaxAggrEdge,
    }

    if model_name not in models:
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {list(models.keys())}"
        )

    return models[model_name](
        config.model_channels, config.feature_dim, config.dropout_ratio
    )


def generate_dataset(args):
    """Generate and save a dataset."""
    dataset = KYNDataset(
        root_data_path=args.root_data_path,
        dataset_naming_convention=args.dataset_type,
        filter_strs=args.filter_strs,
        sample_size=args.sample_size,
        exclude=args.exclude_filter,
        with_edge_features=args.with_edge_features,
    )

    dataset.load_and_transform_graphs()
    dataset.save_dataset(args.output_prefix)
    logger.info(f"Dataset saved with prefix: {args.output_prefix}")


def train_model(args, sweep=False):
    """Train a model using the specified configuration."""
    if sweep:
        # Get parameters from wandb.config during sweeps
        config = KYNConfig(
            learning_rate=wandb.config.learning_rate,
            min_learning_rate=wandb.config.min_learning_rate,
            model_channels=wandb.config.hidden_channels,
            batch_size=wandb.config.batch_size,
            train_data=args.train_data,
            train_labels=args.train_labels,
            model_arch=args.model_name,
            test_data=args.test_data,
            test_labels=args.test_labels,
            circle_loss_m=wandb.config.circle_loss_m,
            circle_loss_gamma=wandb.config.circle_loss_gamma,
            dropout_ratio=wandb.config.dropout_ratio,
            early_stopping_patience=wandb.config.early_stopping_patience,
            early_stopping_delta=wandb.config.early_stopping_delta,
            epochs=wandb.config.epochs,
        )
    else:
        config = KYNConfig(
            learning_rate=args.learning_rate,
            model_channels=args.model_channels,
            feature_dim=args.feature_dim,
            batch_size=args.batch_size,
            train_data=args.train_data,
            train_labels=args.train_labels,
            model_arch=args.model_name,
            test_data=args.test_data,
            test_labels=args.test_labels,
        )

    if (
        args.model_name == "GraphConvInstanceGlobalMaxSmall"
        or args.model_name == "GraphConvInstanceGlobalMaxSmallSoftMaxAggr"
    ):
        config.with_edges = False

    model = get_model(args.model_name, config)
    trainer = KYNTrainer(
        model=model,
        config=config,
        device=args.device,
        log_to_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
    )

    trainer.train(validate_examples=args.validate_examples)

    trainer.save_model()
    if trainer.log_to_wandb:
        wandb.save(f"{trainer.config.exp_uuid}_*.ep{trainer.config.epochs}")

    logger.info(f"Model saved with UUID: {config.exp_uuid}")


def evaluate_model(args):
    """Evaluate a trained model."""
    model = get_model(
        args.model_name,
        KYNConfig(model_channels=args.model_channels, feature_dim=args.feature_dim),
    )
    model.load_state_dict(torch.load(args.model_path))

    evaluator = KYNEvaluator(
        model=model,
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        eval_prefix=args.eval_prefix,
        search_pool_size=args.search_pool_sizes,
        num_search_pools=args.num_search_pools,
        random_seed=args.random_seed,
        requires_edge_feats=args.requires_edge_feats,
    )

    evaluator.evaluate()


def evaluate_vuln_model(args):
    """Evaluate a trained model on vulnerability detection."""
    model = get_model(
        args.model_name,
        KYNConfig(model_channels=args.model_channels, feature_dim=args.feature_dim),
    )
    model.load_state_dict(torch.load(args.model_path))

    evaluator = KYNVulnEvaluator(
        model=model,
        model_name=args.model_name,
        target_data_path=args.target_data_path,
        search_data_paths=args.search_data_paths,
        vulnerable_functions=args.vulnerable_functions,
        device=args.device,
        target_arch=args.target_arch,
        no_metadata=args.no_metadata,
        save_metrics_to_file=not args.no_save_metrics,
    )

    results = evaluator.evaluate()

    # Print summary of results
    for result in results:
        logger.info(f"\nResults for {Path(result['search_data']).name}:")
        logger.info(f"Mean Rank: {result['mean_rank']}")
        logger.info(f"Median Rank: {result['median_rank']}")
        logger.info(f"Mean Similarity: {result['mean_similarity']}")


# Add new sweep command parser
def add_sweep_parser(subparsers):
    sweep_parser = subparsers.add_parser("sweep", help="Run hyperparameter sweep")
    sweep_parser.add_argument(
        "--sweep-config", required=True, help="Path to sweep configuration YAML file"
    )
    sweep_parser.add_argument(
        "--count", type=int, default=20, help="Number of sweep runs to execute"
    )
    sweep_parser.add_argument(
        "--model-name", required=True, help="Name of the model architecture to use"
    )
    sweep_parser.add_argument(
        "--train-data", required=True, help="Path to training data pickle file"
    )
    sweep_parser.add_argument(
        "--train-labels", required=True, help="Path to training labels pickle file"
    )
    sweep_parser.add_argument(
        "--test-data", help="Path to test data pickle file (for validation)"
    )
    sweep_parser.add_argument(
        "--test-labels", help="Path to test labels pickle file (for validation)"
    )
    sweep_parser.add_argument(
        "--feature-dim", type=int, default=6, help="Feature dimension"
    )
    sweep_parser.add_argument(
        "--wandb-project", help="Weights & Biases project name", required=True
    )
    sweep_parser.add_argument(
        "--device", default="cuda", help="Device to use (cuda, cpu, mps)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="KYN - Dataset Generation, Training, and Evaluation CLI"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    add_sweep_parser(subparsers)  # Add this line

    # Dataset generation parser
    dataset_parser = subparsers.add_parser("generate", help="Generate a dataset")
    dataset_parser.add_argument(
        "--root-data-path", required=True, help="Root path containing JSON graph files"
    )
    dataset_parser.add_argument(
        "--dataset-type",
        required=True,
        choices=["cisco", "binkit", "trex", "binarycorp", "custom"],
        help="Dataset naming convention",
    )
    dataset_parser.add_argument(
        "--filter-strs", nargs="*", default=[], help="Strings to filter dataset files"
    )
    dataset_parser.add_argument(
        "--sample-size",
        type=int,
        default=-1,
        help="Number of samples to include (-1 for all)",
    )
    dataset_parser.add_argument(
        "--exclude-filter",
        action="store_true",
        help="Exclude rather than include filter matches",
    )
    dataset_parser.add_argument(
        "--with-edge-features", action="store_true", help="Include edge features"
    )
    dataset_parser.add_argument(
        "--output-prefix", required=True, help="Output file prefix"
    )

    # Training parser
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--model-name", required=True, help="Name of the model architecture to use"
    )
    train_parser.add_argument(
        "--train-data", required=True, help="Path to training data pickle file"
    )
    train_parser.add_argument(
        "--train-labels", required=True, help="Path to training labels pickle file"
    )
    train_parser.add_argument(
        "--learning-rate", type=float, default=5e-4, help="Learning rate"
    )
    train_parser.add_argument(
        "--model-channels", type=int, default=256, help="Number of model channels"
    )
    train_parser.add_argument(
        "--feature-dim", type=int, default=6, help="Feature dimension"
    )
    train_parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    train_parser.add_argument(
        "--device", default="cuda", help="Device to use (cuda, cpu, mps)"
    )
    train_parser.add_argument(
        "--use-wandb", action="store_true", help="Log to Weights & Biases"
    )
    train_parser.add_argument("--wandb-project", help="Weights & Biases project name")
    train_parser.add_argument(
        "--validate-examples",
        action="store_true",
        help="Validate examples during training",
    )
    train_parser.add_argument(
        "--test-data", help="Path to test data pickle file (for validation)"
    )
    train_parser.add_argument(
        "--test-labels", help="Path to test labels pickle file (for validation)"
    )

    # Evaluation parser
    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate a trained model on generic search"
    )
    eval_parser.add_argument(
        "--model-path", required=True, help="Path to the trained model file"
    )
    eval_parser.add_argument(
        "--model-name", required=True, help="Name of the model architecture"
    )
    eval_parser.add_argument(
        "--model-channels", type=int, default=512, help="Number of model channels"
    )
    eval_parser.add_argument(
        "--feature-dim", type=int, default=6, help="Feature dimension"
    )
    eval_parser.add_argument(
        "--dataset-path", required=True, help="Path to evaluation dataset"
    )
    eval_parser.add_argument(
        "--eval-prefix", required=True, help="Prefix for eval results"
    )
    eval_parser.add_argument(
        "--search-pool-sizes",
        type=int,
        nargs="+",
        default=[100, 250, 500, 1000],
        help="Search pool sizes",
    )
    eval_parser.add_argument(
        "--num-search-pools", type=int, default=1000, help="Number of search pools"
    )
    eval_parser.add_argument(
        "--random-seed", type=int, default=1337, help="Random seed"
    )
    eval_parser.add_argument(
        "--requires-edge-feats",
        action="store_true",
        help="Model requires edge features",
    )

    # Vulnerability Evaluation parser
    vuln_eval_parser = subparsers.add_parser(
        "vuln-evaluate", help="Evaluate a trained model on vulnerability detection"
    )

    # Model parameters
    vuln_eval_parser.add_argument(
        "--model-path", required=True, help="Path to the trained model file"
    )
    vuln_eval_parser.add_argument(
        "--model-name", required=True, help="Name of the model architecture"
    )
    vuln_eval_parser.add_argument(
        "--model-channels", type=int, default=256, help="Number of model channels"
    )
    vuln_eval_parser.add_argument(
        "--feature-dim", type=int, default=6, help="Feature dimension"
    )
    vuln_eval_parser.add_argument(
        "--device", default="cuda", help="Device to use (cuda, cpu, mps)"
    )

    # Data paths
    vuln_eval_parser.add_argument(
        "--target-data-path", required=True, help="Path to the target firmware data"
    )
    vuln_eval_parser.add_argument(
        "--search-data-paths", required=True, nargs="+", help="Paths to search datasets"
    )

    # Vulnerability configuration
    vuln_eval_parser.add_argument(
        "--target-arch", required=True, help="Target architecture (e.g., mips32, arm32)"
    )
    vuln_eval_parser.add_argument(
        "--vulnerable-functions",
        required=True,
        nargs="+",
        help="List of vulnerable function names to search for",
    )

    # Additional options
    vuln_eval_parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip metadata processing in graph loading",
    )
    vuln_eval_parser.add_argument(
        "--no-save-metrics", action="store_true", help="Don't save metrics to files"
    )

    args = parser.parse_args()
    logger.level("INFO")
    if args.command == "generate":
        generate_dataset(args)
    elif args.command == "train":
        train_model(args)
    elif args.command == "evaluate":
        evaluate_model(args)
    elif args.command == "vuln-evaluate":
        evaluate_vuln_model(args)
    elif args.command == "sweep":
        with open(args.sweep_config, "r") as f:
            sweep_config = yaml.safe_load(f)

        # Initialize sweep
        sweep_id = wandb.sweep(sweep=sweep_config, project=args.wandb_project)

        # Define sweep training function
        def sweep_train():
            train_args = argparse.Namespace(
                model_name=args.model_name,
                train_data=args.train_data,
                train_labels=args.train_labels,
                test_data=args.test_data,
                test_labels=args.test_labels,
                use_wandb=True,
                wandb_project=args.wandb_project,
                device=args.device,
                validate_examples=False,
            )
            wandb.init()

            # fuse sweep config with KYNConfig to pass to wandb, cheap solution
            _config = KYNConfig(
                learning_rate=wandb.config.learning_rate,
                min_learning_rate=wandb.config.min_learning_rate,
                model_channels=wandb.config.hidden_channels,
                batch_size=wandb.config.batch_size,
                train_data=args.train_data,
                train_labels=args.train_labels,
                model_arch=args.model_name,
                test_data=args.test_data,
                test_labels=args.test_labels,
                circle_loss_m=wandb.config.circle_loss_m,
                circle_loss_gamma=wandb.config.circle_loss_gamma,
            )

            wandb.config.update(_config.to_dict())

            train_model(train_args, sweep=True)

        # Run sweep agent
        wandb.agent(sweep_id, function=sweep_train, count=args.count)
        return
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
