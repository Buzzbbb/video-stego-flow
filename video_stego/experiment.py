"""
experiment.py — 实验管理模块
Experiment save/load manager.

:class:`ExperimentManager` serialises experiment configurations and metric
results to JSON so that experiments can be reproduced and compared later.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional


class ExperimentManager:
    """Save and load steganography experiment records.

    Each record is a dictionary that may contain:

    * ``"id"`` — a unique string identifier
    * ``"timestamp"`` — Unix timestamp of when the record was created
    * ``"params"`` — embedding / attack parameters (arbitrary dict)
    * ``"metrics"`` — objective results (PSNR, SSIM, BER, …)
    * ``"notes"`` — free-text description

    Records are stored as a JSON file at *storage_path*.  Multiple records
    can coexist in the same file.

    Parameters
    ----------
    storage_path:
        Path to the JSON file used for persistence.

    Examples
    --------
    >>> mgr = ExperimentManager("experiments.json")
    >>> exp_id = mgr.save(
    ...     params={"method": "LSB", "lsb_count": 1},
    ...     metrics={"psnr": 52.4, "ber": 0.0},
    ...     notes="baseline LSB experiment",
    ... )
    >>> record = mgr.load(exp_id)
    >>> all_records = mgr.list_records()
    """

    def __init__(self, storage_path: str = "experiments.json") -> None:
        self.storage_path = storage_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        params: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
        notes: str = "",
        experiment_id: Optional[str] = None,
    ) -> str:
        """Persist an experiment record and return its ID.

        Parameters
        ----------
        params:
            Dictionary of experiment parameters (embedding method, attack
            settings, etc.).
        metrics:
            Optional dictionary of objective quality / robustness metrics.
        notes:
            Optional free-text description.
        experiment_id:
            If supplied, use this as the record's ID (must be unique within
            the store).  Otherwise, an ID is generated automatically.

        Returns
        -------
        str
            The ID of the saved record.
        """
        records = self._load_all()
        ts = time.time()
        exp_id = experiment_id or f"exp_{int(ts * 1000)}"
        if any(r["id"] == exp_id for r in records):
            raise ValueError(f"Experiment ID already exists: {exp_id!r}")
        record: Dict[str, Any] = {
            "id": exp_id,
            "timestamp": ts,
            "params": params,
            "metrics": metrics or {},
            "notes": notes,
        }
        records.append(record)
        self._save_all(records)
        return exp_id

    def load(self, experiment_id: str) -> Dict[str, Any]:
        """Retrieve the record with the given *experiment_id*.

        Raises
        ------
        KeyError
            If no record with the given ID exists.
        """
        for record in self._load_all():
            if record["id"] == experiment_id:
                return record
        raise KeyError(f"No experiment found with id: {experiment_id!r}")

    def list_records(self) -> List[Dict[str, Any]]:
        """Return all stored experiment records, sorted by timestamp."""
        records = self._load_all()
        return sorted(records, key=lambda r: r.get("timestamp", 0))

    def delete(self, experiment_id: str) -> None:
        """Remove the record with the given *experiment_id*.

        Raises
        ------
        KeyError
            If no record with the given ID exists.
        """
        records = self._load_all()
        new_records = [r for r in records if r["id"] != experiment_id]
        if len(new_records) == len(records):
            raise KeyError(f"No experiment found with id: {experiment_id!r}")
        self._save_all(new_records)

    def update_metrics(
        self, experiment_id: str, metrics: Dict[str, Any]
    ) -> None:
        """Merge *metrics* into the record identified by *experiment_id*."""
        records = self._load_all()
        for record in records:
            if record["id"] == experiment_id:
                record["metrics"].update(metrics)
                self._save_all(records)
                return
        raise KeyError(f"No experiment found with id: {experiment_id!r}")

    def summary(self) -> str:
        """Return a human-readable summary of all stored experiments."""
        records = self.list_records()
        if not records:
            return "No experiments stored."
        lines = [f"{'ID':<30}  {'Timestamp':<22}  Notes"]
        lines.append("-" * 80)
        for r in records:
            ts = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(r.get("timestamp", 0))
            )
            lines.append(f"{r['id']:<30}  {ts:<22}  {r.get('notes', '')}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_all(self) -> List[Dict[str, Any]]:
        if not os.path.isfile(self.storage_path):
            return []
        with open(self.storage_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                return []
        return data if isinstance(data, list) else []

    def _save_all(self, records: List[Dict[str, Any]]) -> None:
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
