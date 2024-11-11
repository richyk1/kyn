import argparse
from pathlib import Path

import torch
from loguru import logger

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
from kyn.eval import KYNEvaluator, KYNVulnEvaluator


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
        dataset_naming_convetion=args.dataset_type,
        filter_strs=args.filter_strs,
        sample_size=args.sample_size,
        exclude=args.exclude_filter,
        with_edge_features=args.with_edge_features,
    )

    dataset.load_and_transform_graphs()
    dataset.save_dataset(args.output_prefix)
    logger.info(f"Dataset saved with prefix: {args.output_prefix}")


def train_model(args):
    """Train a model using the specified configuration."""
    config = KYNConfig(
        learning_rate=args.learning_rate,
        model_channels=args.model_channels,
        feature_dim=args.feature_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_data=args.train_data,
        train_labels=args.train_labels,
        model_arch=args.model_name,
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
        eval_prefix=args.experiment_prefix,
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


def main():
    parser = argparse.ArgumentParser(
        description="KYN - Dataset Generation, Training, and Evaluation CLI"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Dataset generation parser
    dataset_parser = subparsers.add_parser("generate", help="Generate a dataset")
    dataset_parser.add_argument(
        "--root-data-path", required=True, help="Root path containing JSON graph files"
    )
    dataset_parser.add_argument(
        "--dataset-type",
        required=True,
        choices=["cisco", "binkit", "trex", "binarycorp"],
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
    train_parser.add_argument(
        "--epochs", type=int, default=350, help="Number of epochs"
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
        "--model-channels", type=int, default=256, help="Number of model channels"
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
        default=[100],
        help="Search pool sizes",
    )
    eval_parser.add_argument(
        "--num-search-pools", type=int, default=500, help="Number of search pools"
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
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
