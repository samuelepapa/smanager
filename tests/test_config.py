"""Tests for configuration handling."""

import tempfile
from pathlib import Path

from smanager.config import SManagerConfig


def test_config_without_smanager_dir():
    """Test config when no .smanager directory exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir).resolve()
        script_path = tmpdir / "script.py"
        script_path.touch()

        config = SManagerConfig(script_path)

        assert config.config_dir is None
        assert config.preamble == ""
        assert config.defaults == {}


def test_config_with_smanager_dir():
    """Test config discovery when .smanager exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir).resolve()

        # Create .smanager directory
        smanager_dir = tmpdir / ".smanager"
        smanager_dir.mkdir()

        # Create preamble
        preamble_path = smanager_dir / "preamble.sh"
        preamble_path.write_text("source activate myenv\n")

        # Create config
        config_path = smanager_dir / "config.yaml"
        config_path.write_text("gpus: 4\nmemory: 32G\n")

        # Create script
        script_path = tmpdir / "script.py"
        script_path.touch()

        config = SManagerConfig(script_path)

        assert config.config_dir == smanager_dir
        assert "source activate myenv" in config.preamble
        assert config.defaults.get("gpus") == 4
        assert config.defaults.get("memory") == "32G"


def test_config_discovery_in_parent():
    """Test that config is discovered in parent directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir).resolve()

        # Create .smanager in root
        smanager_dir = tmpdir / ".smanager"
        smanager_dir.mkdir()
        (smanager_dir / "preamble.sh").write_text("# preamble\n")

        # Create nested script directory
        script_dir = tmpdir / "experiments" / "models"
        script_dir.mkdir(parents=True)
        script_path = script_dir / "train.py"
        script_path.touch()

        config = SManagerConfig(script_path)

        assert config.config_dir == smanager_dir


def test_get_script_dir():
    """Test script directory creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir).resolve()

        # Create .smanager
        smanager_dir = tmpdir / ".smanager"
        smanager_dir.mkdir()

        script_path = tmpdir / "script.py"
        script_path.touch()

        config = SManagerConfig(script_path)
        script_dir = config.get_script_dir()

        assert script_dir.exists()
        assert script_dir == smanager_dir / "scripts"
