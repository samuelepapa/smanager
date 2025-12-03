"""Tests for sweep handling."""

import tempfile
from pathlib import Path

import pytest

from smanager.sweep import Sweep, load_sweep_function


def create_sweep_file(tmpdir: Path) -> Path:
    """Create a test sweep file."""
    sweep_file = tmpdir / "sweeps.py"
    sweep_file.write_text(
        '''
def simple_sweep():
    """Simple test sweep."""
    for lr in [0.1, 0.01]:
        yield {"lr": lr}


def grid_sweep():
    """Grid search sweep."""
    for lr in [0.1, 0.01]:
        for batch_size in [32, 64]:
            yield {"lr": lr, "batch_size": batch_size}


def invalid_sweep():
    """Sweep that yields non-dict."""
    yield "not a dict"
'''
    )
    return sweep_file


def test_load_sweep_function():
    """Test loading sweep function from file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir).resolve()
        sweep_file = create_sweep_file(tmpdir)

        func = load_sweep_function(str(sweep_file), "simple_sweep")

        assert callable(func)
        results = list(func())
        assert len(results) == 2
        assert results[0] == {"lr": 0.1}
        assert results[1] == {"lr": 0.01}


def test_load_sweep_function_not_found():
    """Test error when function doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir).resolve()
        sweep_file = create_sweep_file(tmpdir)

        with pytest.raises(AttributeError):
            load_sweep_function(str(sweep_file), "nonexistent_sweep")


def test_load_sweep_function_file_not_found():
    """Test error when file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_sweep_function("/nonexistent/path.py", "sweep")


def test_sweep_generate_jobs():
    """Test generating jobs from sweep."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir).resolve()
        script_path = tmpdir / "train.py"
        script_path.write_text("print('train')")
        sweep_file = create_sweep_file(tmpdir)

        sweep_obj = Sweep(
            script_path=str(script_path),
            sweep_file=str(sweep_file),
            sweep_function="grid_sweep",
            gpus=4,
        )

        jobs = sweep_obj.generate_jobs()

        assert len(jobs) == 4  # 2 lr * 2 batch_size
        assert all(j.gpus == 4 for j in jobs)

        # Check params are stored
        params = [j.sweep_params for j in jobs]
        assert {"lr": 0.1, "batch_size": 32} in params
        assert {"lr": 0.1, "batch_size": 64} in params
        assert {"lr": 0.01, "batch_size": 32} in params
        assert {"lr": 0.01, "batch_size": 64} in params


def test_sweep_with_base_args():
    """Test sweep with base arguments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir).resolve()
        script_path = tmpdir / "train.py"
        script_path.write_text("print('train')")
        sweep_file = create_sweep_file(tmpdir)

        sweep_obj = Sweep(
            script_path=str(script_path),
            sweep_file=str(sweep_file),
            sweep_function="simple_sweep",
            base_args=["--epochs", "100"],
        )

        jobs = sweep_obj.generate_jobs()

        for job in jobs:
            assert "--epochs" in job.script_args
            assert "100" in job.script_args


def test_sweep_invalid_yield():
    """Test error when sweep yields non-dict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir).resolve()
        script_path = tmpdir / "train.py"
        script_path.write_text("print('train')")
        sweep_file = create_sweep_file(tmpdir)

        sweep_obj = Sweep(
            script_path=str(script_path),
            sweep_file=str(sweep_file),
            sweep_function="invalid_sweep",
        )

        with pytest.raises(TypeError, match="must yield dictionaries"):
            sweep_obj.generate_jobs()


def test_sweep_save_scripts():
    """Test saving sweep scripts and mapping."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir).resolve()
        script_path = tmpdir / "train.py"
        script_path.write_text("print('train')")
        sweep_file = create_sweep_file(tmpdir)

        # Create .smanager
        smanager_dir = tmpdir / ".smanager"
        smanager_dir.mkdir()

        sweep_obj = Sweep(
            script_path=str(script_path),
            sweep_file=str(sweep_file),
            sweep_function="simple_sweep",
        )

        paths = sweep_obj.save_scripts()

        assert len(paths) == 2
        assert all(p.exists() for p in paths)
        assert all(p.suffix == ".sbatch" for p in paths)

        # Check sweep.json was created
        sweep_json = sweep_obj.sweep_dir / "sweep.json"
        assert sweep_json.exists()

        # Check logs directory was created
        logs_dir = sweep_obj.sweep_dir / "logs"
        assert logs_dir.exists()
