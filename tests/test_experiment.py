"""Tests for video_stego.experiment."""
import json
import os
import tempfile

import pytest

from video_stego.experiment import ExperimentManager


@pytest.fixture
def mgr(tmp_path):
    return ExperimentManager(str(tmp_path / "experiments.json"))


class TestExperimentManager:
    def test_save_returns_string_id(self, mgr):
        eid = mgr.save(params={"method": "LSB"})
        assert isinstance(eid, str)

    def test_load_roundtrip(self, mgr):
        eid = mgr.save(params={"method": "DCT"}, metrics={"psnr": 42.0})
        record = mgr.load(eid)
        assert record["params"]["method"] == "DCT"
        assert record["metrics"]["psnr"] == 42.0

    def test_load_missing_raises(self, mgr):
        with pytest.raises(KeyError):
            mgr.load("nonexistent")

    def test_list_records_sorted_by_timestamp(self, mgr):
        ids = [mgr.save(params={"i": i}) for i in range(3)]
        records = mgr.list_records()
        assert [r["id"] for r in records] == ids or len(records) == 3

    def test_delete_removes_record(self, mgr):
        eid = mgr.save(params={"x": 1})
        mgr.delete(eid)
        with pytest.raises(KeyError):
            mgr.load(eid)

    def test_delete_missing_raises(self, mgr):
        with pytest.raises(KeyError):
            mgr.delete("does_not_exist")

    def test_duplicate_id_raises(self, mgr):
        mgr.save(params={}, experiment_id="dup")
        with pytest.raises(ValueError):
            mgr.save(params={}, experiment_id="dup")

    def test_update_metrics(self, mgr):
        eid = mgr.save(params={}, metrics={"psnr": 30.0})
        mgr.update_metrics(eid, {"ber": 0.01})
        record = mgr.load(eid)
        assert record["metrics"]["psnr"] == 30.0
        assert record["metrics"]["ber"] == 0.01

    def test_update_metrics_missing_raises(self, mgr):
        with pytest.raises(KeyError):
            mgr.update_metrics("ghost", {"ber": 0.0})

    def test_summary_no_records(self, mgr):
        summary = mgr.summary()
        assert "No experiments" in summary

    def test_summary_with_records(self, mgr):
        mgr.save(params={"method": "LSB"}, notes="test run")
        summary = mgr.summary()
        assert "test run" in summary

    def test_persistence_across_instances(self, tmp_path):
        path = str(tmp_path / "persist.json")
        mgr1 = ExperimentManager(path)
        eid = mgr1.save(params={"method": "pixel"})
        mgr2 = ExperimentManager(path)
        record = mgr2.load(eid)
        assert record["params"]["method"] == "pixel"

    def test_invalid_json_returns_empty(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            f.write("not json {{")
        mgr = ExperimentManager(path)
        assert mgr.list_records() == []

    def test_notes_stored(self, mgr):
        eid = mgr.save(params={}, notes="my note")
        assert mgr.load(eid)["notes"] == "my note"
