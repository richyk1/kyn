import pytest
from kyn.dataset import KYNDataset
import os

TEST_DATA_ROOT = "tests/test-data/"
CISCO_TEST_DATA = os.path.join(TEST_DATA_ROOT, "cisco")
TREX_TEST_DATA = os.path.join(TEST_DATA_ROOT, "trex")
BINARYCORP_TEST_DATA = os.path.join(TEST_DATA_ROOT, "binarycorp")


def test_dataset_init():
    """Test dataset init with loading all data (i.e sample_size set to -1)"""
    data = KYNDataset(CISCO_TEST_DATA, dataset_naming_convetion="cisco")

    assert len(data.file_paths) == 28
    assert isinstance(data, KYNDataset)
    assert isinstance(data.labels, list)
    assert isinstance(data.graphs, list)
    assert data.dataset_naming_convetion == "cisco"


@pytest.mark.parametrize("sample_size, expected_size", [(10, 10), (20, 20), (1, 1)])
def test_dataset_sample_size_defined(sample_size, expected_size):
    """Test dataset init with a sample size set"""
    data = KYNDataset(
        CISCO_TEST_DATA, dataset_naming_convetion="cisco", sample_size=sample_size
    )
    assert len(data.file_paths) == expected_size


def test_dataset_init_empty_folder():
    """Test dataset init when the dataset folder provided is empty"""
    with pytest.raises(ValueError):
        _ = KYNDataset("tests/no_data", dataset_naming_convetion="cisco")


def test_dataset_init_incorrect_naming_convetion():
    """Test dataset init when an invalid dataset naming convetion is provided"""
    with pytest.raises(ValueError):
        _ = KYNDataset(
            CISCO_TEST_DATA, dataset_naming_convetion="random", sample_size=10
        )


def test_dataset_init_with_filter_string():
    """Test dataset init when a filter string is provided"""
    dataset = KYNDataset(
        CISCO_TEST_DATA, dataset_naming_convetion="cisco", filter_strs=["_set_"]
    )
    assert len(dataset.file_paths) == 2


@pytest.mark.parametrize(
    "idx, expected",
    [(0, "afalg.sosym.ERR_AFALG_error"), (5, "afalg.sosym.afalg_chk_platform")],
)
def test_binary_func_id_extraction_cisco(idx, expected):
    """Test binary func id extraction when provided with a cisco style filepath"""
    dataset = KYNDataset(CISCO_TEST_DATA, dataset_naming_convetion="cisco")
    dataset.file_paths = sorted(dataset.file_paths)

    target = dataset.file_paths[idx]
    ret = dataset.get_cisco_talos_binary_func_id(target)
    assert ret == expected


@pytest.mark.parametrize(
    "idx, expected", [(0, "elfeditentry0"), (5, "elfeditsym.adjust_relative_path")]
)
def test_binary_func_id_extraction_trex(idx, expected):
    """Test binary func id extraction when provided with a trex style filepath"""
    dataset = KYNDataset(TREX_TEST_DATA, dataset_naming_convetion="trex")
    dataset.file_paths = sorted(dataset.file_paths)

    target = dataset.file_paths[idx]
    ret = dataset.get_trex_binary_func_id(target)
    assert ret == expected


@pytest.mark.parametrize(
    "idx, expected",
    [
        (0, "mod_proxy_http.sosym.add_cl"),
        (5, "mod_proxy_http.sosym.proxy_http_async_cb"),
    ],
)
def test_binary_func_id_extraction_binarycorp(idx, expected):
    """Test binary func id extraction when provided with a binarycorp style filepath"""
    dataset = KYNDataset(BINARYCORP_TEST_DATA, dataset_naming_convetion="binarycorp")
    dataset.file_paths = sorted(dataset.file_paths)

    target = dataset.file_paths[idx]
    ret = dataset.get_binarycorp_binary_func_id(target)
    assert ret == expected
