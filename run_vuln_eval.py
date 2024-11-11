#!/usr/bin/env python3
from kyn.networks import GraphConvInstanceGlobalMaxSmallSoftMaxAggrEdge
import os
import torch
from kyn.eval import KYNVulnEvaluator
import argparse
from pathlib import Path
from loguru import logger

# Known vulnerable functions for different devices
DEVICE_VULNS = {
    "tplink": [
        "CMS_decrypt",
        "PKCS7_dataDecode",
        "BN_bn2dec",
        "EVP_EncodeUpdate",
        "BN_dec2bn",
        "BN_hex2bn"
    ],
    "netgear": [
        "CMS_decrypt",
        "PKCS7_dataDecode",
        "MDC2_Update",
        "BN_bn2dec"
    ]
}

# Standard search paths relative to data root
SEARCH_ARCHS = [
    "mips32",
    "x64",
    "x86",
    "arm32",
    "ppc32",
    "riscv32"
]


def get_search_paths(data_root: str) -> list[str]:
    """Generate standard search paths for different architectures."""
    paths = []
    for arch in SEARCH_ARCHS:
        # Handle special case for riscv32 which uses 1.0.0d
        version = "1.0.0d" if arch == "riscv32" else "1.0.2d"
        path = os.path.join(
            data_root,
            f"libcrypto.so.1.0.0_openssl_{version}_{arch}_cg-onehopcgcallers-meta"
        )
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description='Evaluate vulnerability detection across different architectures')

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to the model checkpoint file'
    )

    parser.add_argument(
        '--data-root',
        type=str,
        required=True,
        help='Root directory containing the vulnerability dataset'
    )

    parser.add_argument(
        '--target',
        choices=['tplink', 'netgear'],
        required=True,
        help='Target device to evaluate (tplink or netgear)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root directory not found: {args.data_root}")

    # Load model
    logger.info(f"Loading model from {args.model}")
    model = GraphConvInstanceGlobalMaxSmallSoftMaxAggrEdge(256, 6)
    model_dict = torch.load(args.model)
    model.load_state_dict(model_dict)
    model.eval()

    # Set up target data path based on device
    target_paths = {
        "tplink": "libcrypto.so.1.0.0_TP-Link_Deco-M4_1.0.2d_mips32_cg-onehopcgcallers-meta",
        "netgear": "libcrypto.so.1.0.0_NETGEAR_R7000_1.0.2h_arm32_cg-onehopcgcallers-meta"
    }

    target_data = os.path.join(data_root, target_paths[args.target])
    if not os.path.exists(target_data):
        raise FileNotFoundError(f"Target data directory not found: {target_data}")

    # Get search paths
    search_paths = get_search_paths(str(data_root))
    valid_paths = [p for p in search_paths if os.path.exists(p)]

    if not valid_paths:
        raise FileNotFoundError("No valid search paths found in data root")

    logger.info(f"Evaluating {args.target} device against {len(valid_paths)} architecture variants")

    # Create evaluator instance
    evaluator = KYNVulnEvaluator(
        model=model,
        model_name=Path(args.model).stem,
        target_data_path=target_data,
        search_data_paths=valid_paths,
        vulnerable_functions=DEVICE_VULNS[args.target],
        target_arch="mips32" if args.target == "tplink" else "arm32"
    )

    # Run evaluation
    results = evaluator.evaluate()

    # Print summary
    logger.info("\nEvaluation Summary:")
    for result in results:
        logger.info(f"\nResults for {Path(result['search_data']).name}")
        logger.info(f"Mean Rank: {result['mean_rank']}")
        logger.info(f"Median Rank: {result['median_rank']}")
        logger.info(f"Mean Similarity: {result['mean_similarity']}")


if __name__ == "__main__":
    main()