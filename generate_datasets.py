from kyn.dataset import KYNDataset
from loguru import logger

logger.info("This script assumes you have locally processed the data with bin2ml. Every options "
            "apart from BINARYCORP_WITH_DUPS assumes you have already run the de-duplication functionality "
            "bin2ml.")
# Train Data
CISCO_DATASET_1_DATA_RAW = None

# Eval Data
CISCO_DATASET_2_DATA_RAW = None
BINKIT_NOINLINE_DATA_RAW = None
BINKIT_NORMAL_DATA_RAW = None
BINARYCORP_DATA_RAW = None
BINARYCORP_WITH_DUPS_RAW = None

logger.info("Generating datasets...")
if CISCO_DATASET_1_DATA_RAW is not None:
    logger.info("Starting to process Cisco Dataset 1")
    dataset = KYNDataset(
        root_data_path=CISCO_DATASET_1_DATA_RAW,
        dataset_naming_convetion="cisco",
        filter_strs=["nmap", "z3", "nping", "ncat"],
        exclude=True,
        sample_size=256625,
    )

    dataset.load_and_transform_graphs()
    dataset.save_dataset("cisco-d1-train-callers-edge-between")

    dataset = KYNDataset(
        root_data_path=CISCO_DATASET_1_DATA_RAW,
        dataset_naming_convetion="cisco",
        filter_strs=["nmap", "z3", "nping", "ncat"],
        sample_size=522003,
    )

    dataset.load_and_transform_graphs()
    dataset.save_dataset("cisco-d1-test-callers-edge-between")
else:
    logger.warning("The value of CISCO_DATASET_1_DATA_RAW is None. Skipping...")


if CISCO_DATASET_2_DATA_RAW is not None:
    logger.info("Starting to process Cisco Dataset 2")
    dataset = KYNDataset(
        root_data_path=CISCO_DATASET_2_DATA_RAW,
        dataset_naming_convetion="trex",
    )

    dataset.load_and_transform_graphs()
    dataset.save_dataset("cisco-d2-test-callers-edge-between")
else:
    logger.warning(f"The value of CISCO_DATASET_2_DATA_RAW is None. Skipping...")


if BINKIT_NOINLINE_DATA_RAW is not None:
    logger.info("Starting to process Binkit No-inline")
    dataset = KYNDataset(root_data_path=BINKIT_NOINLINE_DATA_RAW,
                         sample_size=1000000,
                         dataset_naming_convetion="binkit")

    dataset.load_and_transform_graphs()
    dataset.save_dataset("binkit-no-inline-test-callers-edge-between")
else:
    logger.warning(f"The value of BINKIT_NOINLINE_DATA_RAW is None. Skipping...")

if BINKIT_NORMAL_DATA_RAW is not None:
    logger.info("Starting to process Binkit Normal")
    dataset = KYNDataset(root_data_path=BINKIT_NORMAL_DATA_RAW,
                         sample_size=1000000,
                         dataset_naming_convetion="binkit")

    dataset.load_and_transform_graphs()
    dataset.save_dataset("binkit-normal-test-callers-edge-between")
else:
    logger.warning(f"The value of BINKIT_NORMAL_DATA_RAW is None. Skipping...")


if BINARYCORP_DATA_RAW is not None:
    logger.info("Starting to process BinaryCorp (No duplicates)")
    dataset = KYNDataset(root_data_path=BINARYCORP_DATA_RAW,
                         dataset_naming_convetion="binarycorp")

    dataset.load_and_transform_graphs()
    dataset.save_dataset("binarycorp-3m-test")
else:
    logger.warning(f"The value of BINARYCORP_DATA_RAW is None. Skipping...")

if BINARYCORP_WITH_DUPS_RAW is not None:
    logger.info("Starting to process BinaryCorp (with duplicates")
    dataset = KYNDataset(root_data_path=BINARYCORP_WITH_DUPS_RAW,
                         dataset_naming_convetion="binarycorp")

    dataset.load_and_transform_graphs()
    dataset.save_dataset("binarycorp-3m-test-with-dups")
else:
    logger.warning(f"The value of BINARYCORP_WITH_DUPS_RAW is None. Skipping...")
