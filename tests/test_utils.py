"""Tests for shard addressing and the fail-loud data-integrity guards."""

import pytest

from fastdetector import utils


class _FakeDataset:
    """Stands in for a datasets.Dataset for filter-condition tests."""

    def __init__(self, rows):
        self.rows = rows
        self.column_names = sorted(rows[0].keys()) if rows else []

    def filter(self, fn, **kwargs):
        return _FakeDataset([r for r in self.rows if fn(r)])

    def __len__(self):
        return len(self.rows)


class _Cond:
    def __init__(self, column, operator, value):
        self.column = column
        self.operator = operator
        self.value = value


@pytest.fixture
def hub(monkeypatch):
    """Fake Hub: records which (repo, config) pair was requested."""
    state = {"configs": [], "loaded": None}

    def fake_config_names(name):
        if isinstance(state["configs"], Exception):
            raise state["configs"]
        return state["configs"]

    def fake_load_dataset(repo_id, split="train", name=None):
        state["loaded"] = name
        return _FakeDataset([{"a": 1}])

    monkeypatch.setattr(utils, "get_dataset_config_names", fake_config_names)
    monkeypatch.setattr(utils, "load_dataset", fake_load_dataset)
    return state


def test_shard_resolved_by_name_not_position(hub):
    """shard_10 must load shard_10 even when it is not the 10th config.

    Resolving positionally is what let a job read one shard and write its
    results over another.
    """
    hub["configs"] = ["shard_0", "shard_1", "shard_10", "shard_2"]
    utils.load_dataset_auto_shard("repo/ds", subset_index=10)
    assert hub["loaded"] == "shard_10"

    utils.load_dataset_auto_shard("repo/ds", subset_index=2)
    assert hub["loaded"] == "shard_2"


def test_read_and_write_config_names_agree():
    """Readers and writers must derive the config name the same way."""
    hub_configs = [utils.shard_config_name(i) for i in range(3)]
    assert hub_configs == ["shard_0", "shard_1", "shard_2"]


def test_missing_shard_raises_instead_of_silent_default(hub):
    """Out-of-range used to fall back to the default config, so every
    machine in a batched run silently processed identical rows."""
    hub["configs"] = ["shard_0", "shard_1"]
    with pytest.raises(RuntimeError, match="shard_5"):
        utils.load_dataset_auto_shard("repo/ds", subset_index=5)


def test_single_config_dataset_allowed_only_for_batch_zero(hub):
    """filter.py without output_shards produces one 'default' config."""
    hub["configs"] = ["default"]
    utils.load_dataset_auto_shard("repo/ds", subset_index=0)
    assert hub["loaded"] == "default"

    with pytest.raises(RuntimeError):
        utils.load_dataset_auto_shard("repo/ds", subset_index=1)


def test_config_listing_failure_raises(hub):
    hub["configs"] = ConnectionError("hub unreachable")
    with pytest.raises(RuntimeError, match="Could not list configs"):
        utils.load_dataset_auto_shard("repo/ds", subset_index=0)


def test_subset_index_none_loads_default(hub):
    hub["configs"] = ["shard_0"]
    utils.load_dataset_auto_shard("repo/ds", subset_index=None)
    assert hub["loaded"] is None


def test_filter_on_unknown_column_raises():
    """A typo'd column with AND would otherwise drop every row silently."""
    ds = _FakeDataset([{"score": 0.9}, {"score": 0.1}])
    with pytest.raises(KeyError, match="scorre"):
        utils.apply_filter_conditions(ds, [_Cond("scorre", ">", 0.5)], "AND")


def test_filter_still_applies_normally():
    ds = _FakeDataset([{"score": 0.9}, {"score": 0.1}])
    out = utils.apply_filter_conditions(ds, [_Cond("score", ">", 0.5)], "AND")
    assert len(out) == 1


def test_filter_treats_null_cell_as_non_match():
    ds = _FakeDataset([{"score": None}, {"score": 0.9}])
    out = utils.apply_filter_conditions(ds, [_Cond("score", ">", 0.5)], "AND")
    assert len(out) == 1
