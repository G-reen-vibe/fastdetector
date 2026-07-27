"""Regression tests for the visualization layer.

These guard the two failure modes that made ``scripts/analysis.py``
unrunnable: a circular import between ``plotting`` and ``auto_visualizer``,
and markdown tables emitted with escaped ``\\n`` instead of real newlines.
"""

import importlib
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest


class _Series:
    """Minimal object satisfying the PlotSeries protocol."""

    def __init__(self, name, arr):
        self.name = name
        self.arr = np.asarray(arr, dtype=float)


@pytest.mark.parametrize(
    "module",
    [
        "fastdetector.visualization.plotting",
        "fastdetector.visualization.auto_visualizer",
    ],
)
def test_module_imports_standalone(module):
    """Each module must import on its own, in a fresh interpreter.

    Importing them in-process would let an earlier test mask a cycle via
    ``sys.modules``, so this shells out.
    """
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"{module} failed to import:\n{result.stderr}"
    )


def test_plotting_does_not_import_auto_visualizer():
    """plotting is the lower layer; depending upward reintroduces the cycle."""
    source = importlib.import_module("fastdetector.visualization.plotting").__file__
    with open(source, encoding="utf-8") as f:
        text = f.read()
    assert "from fastdetector.visualization.auto_visualizer import" not in text


def test_generate_table_emits_real_newlines():
    from fastdetector.visualization.plotting import generate_table

    class _Cell:
        values = {"acc": 0.5}

    table, _ = generate_table(
        rows=[{"name": "row_a", "cells": [_Cell()]}],
        columns=[{"header": "Accuracy", "wrapper_idx": 0, "stat": "acc"}],
    )

    assert "\\n" not in table, "table contains the literal two-character escape"
    assert table.count("\n") >= 3, "expected header, separator and body rows"
    header, sep, body = table.split("\n")[:3]
    assert header.startswith("| Name |")
    assert sep.startswith("|---|")
    assert body.startswith("| row_a |")


def test_get_histogram_labels_series_by_name():
    """Histograms label series with ``.name``; ``.label`` does not exist."""
    from fastdetector.visualization.plotting import get_histogram

    png = get_histogram([_Series("human", [0.1, 0.2, 0.3])], title="t")
    assert png.startswith(b"\x89PNG")


def test_get_histogram_handles_constant_and_empty_input():
    from fastdetector.visualization.plotting import get_histogram

    assert get_histogram([_Series("flat", [0.5, 0.5])], title="t").startswith(b"\x89PNG")
    assert get_histogram([_Series("none", [])], title="t").startswith(b"\x89PNG")
