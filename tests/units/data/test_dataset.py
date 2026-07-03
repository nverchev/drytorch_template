"""Tests for the data modules."""

from src.data.dataset import get_databuilder, get_datasets
from src.data.databuilder import DataBuilder
from src.data.split import Partitions
from src.data.synthetic_dataset import SyntheticDataset
from src.config import get_config_all


class TestDataBuilder:
    """Test DataBuilder factory and data structure instantiation."""

    def test_get_databuilder_returns_instance(self) -> None:
        """Test get_databuilder returns a DataBuilder instance."""
        builder = get_databuilder()
        assert isinstance(builder, DataBuilder)
        return

    def test_get_datasets_returns_splits(self) -> None:
        """Test get_datasets builds training and evaluation datasets."""
        train_ds, eval_ds = get_datasets()
        assert len(train_ds) > 0
        assert len(eval_ds) > 0
        return


class TestSyntheticDataset:
    """Test SyntheticDataset initialization and sample indexing."""

    def test_synthetic_dataset_sample_shape(self) -> None:
        """Test SyntheticDataset item shape and types."""
        cfg = get_config_all()
        ds = SyntheticDataset()
        split = ds.get_split(Partitions.train)
        assert len(split) == cfg.data.dataset.n_train_samples
        inputs, targets = split[0]
        assert inputs.features.shape[-1] == cfg.data.dataset.in_features
        assert targets.label is not None
        return
