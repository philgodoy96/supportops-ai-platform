"""Tests for the root application package."""

import supportops


def test_package_exposes_project_version() -> None:
    assert supportops.__version__ == "0.1.0"
