#!/usr/bin/env python3
from kyn.networks import GraphConvInstanceGlobalMaxSmallSoftMaxAggrEdge
import torch
from kyn.eval import KYNEvaluator
import argparse
from pathlib import Path
from loguru import logger
import os

def main():
    parser = argparse.ArgumentParser(description='Evaluate model performance on similarity search tasks')

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to the model checkpoint file'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to the evaluation dataset (without -graphs.pickle or -labels.pickle suffix)'
    )

    parser.add_argument(
        '--prefix',
        type=str,
        required=True,
        help='Prefix for the experiment (e.g., binkit-normal, cisco-d2-test)'
    )

    parser.add_argument(
        '--pools',
        type=int,
        default=500,
        help='Number of search pools to evaluate (default: 500)'
    )

    parser.add_argument(
        '--pool-sizes',
        type=int,
        nargs='+',
        default=[100, 250, 1000, 10000],
        help='List of pool sizes to evaluate (default: 100 250 1000 10000)'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default='cuda',
        help='Device to run evaluation on (default: cuda)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=1337,
        help='Random seed for reproducibility (default: 1337)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    dataset_base = Path(args.dataset)
    if not (dataset_base.with_suffix('.pickle').exists() or
            Path(str(dataset_base) + '-graphs.pickle').exists()):
        raise FileNotFoundError(
            f"Dataset files not found: {dataset_base}-graphs.pickle or {dataset_base}-labels.pickle"
        )

    # Load model
    logger.info(f"Loading model from {args.model}")
    model = GraphConvInstanceGlobalMaxSmallSoftMaxAggrEdge(256, 6)
    model_dict = torch.load(args.model)
    model.load_state_dict(model_dict)
    model.eval()

    # Create evaluator instance
    logger.info(f"Initializing evaluator with {len(args.pool_sizes)} pool sizes: {args.pool_sizes}")
    logger.info(f"Using {args.pools} search pools")

    evaluator = KYNEvaluator(
        model=model,
        model_name=Path(args.model).stem,
        dataset_path=str(dataset_base),
        eval_prefix=args.prefix,
        device=args.device,
        search_pool_size=args.pool_sizes,
        num_search_pools=args.pools,
        random_seed=args.seed
    )

    # Run evaluation
    evaluator.evaluate()

    # Results are automatically saved to files, but let's also print a summary
    logger.info("\nEvaluation Summary:")
    for pool_size, metrics in evaluator.metric_dicts:
        logger.info(f"\nResults for pool size {pool_size}:")
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")


def format_dataset_name(dataset_path: str) -> str:
    """Format dataset name for display purposes."""
    return Path(dataset_path).stem.replace('-callers-edge-between', '')


if __name__ == "__main__":
    main()