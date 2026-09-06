# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Release metadata consistency."""
from pathlib import Path
import tomllib

from mathkernel import MathKernel


ROOT = Path(__file__).parents[1]


def test_release_version_package_metadata_and_typed_marker_are_consistent():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["version"] == MathKernel().capabilities()["version"] == __import__("mathkernel").__version__
    assert project["requires-python"] == ">=3.11"
    assert "Programming Language :: Python :: 3.13" in project["classifiers"]
    assert (ROOT / "src/mathkernel/py.typed").is_file()
    package_data = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["tool"]["setuptools"]["package-data"]
    assert package_data["mathkernel_viz"] == ["assets/*"]
    assert package_data["mathkernel_multimodal"] == ["assets/*"]

