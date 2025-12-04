"""Tests for local sweep handling."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from smanager.local import LocalSweep


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


def create_preamble(smanager_dir: Path) -> None:
    """Create a test preamble file."""
    preamble = smanager_dir / "preamble.sh"
    preamble.write_text(
        """# Test preamble
source ~/miniconda3/bin/activate
conda activate myenv
"""
    )


def test_local_sweep_generate_param_sets():
    """Test generating parameter sets from sweep."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir).resolve()
        script_path = tmpdir / "train.py"
        script_path.write_text("print('train')")
        sweep_file = create_sweep_file(tmpdir)

        local_obj = LocalSweep(
            script_path=str(script_path),
            sweep_file=str(sweep_file),
            sweep_function="grid_sweep",
            workers=2,
            gpus="0,1",
        )

        params = local_obj.generate_param_sets()

        assert len(params) == 4  # 2 lr * 2 batch_size
        assert {"lr": 0.1, "batch_size": 32} in params
        assert {"lr": 0.1, "batch_size": 64} in params
        assert {"lr": 0.01, "batch_size": 32} in params
        assert {"lr": 0.01, "batch_size": 64} in params


def test_local_sweep_with_base_args():
    """Test local sweep with base arguments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir).resolve()
        script_path = tmpdir / "train.py"
        script_path.write_text("print('train')")
        sweep_file = create_sweep_file(tmpdir)

        local_obj = LocalSweep(
            script_path=str(script_path),
            sweep_file=str(sweep_file),
            sweep_function="simple_sweep",
            base_args=["--epochs", "100"],
            workers=2,
        )

        local_obj.generate_param_sets()
        scripts = local_obj.save_scripts()

        # Check that base args are in the generated scripts
        for script_path in scripts:
            content = script_path.read_text()
            assert "--epochs" in content
            assert "100" in content


def test_local_sweep_invalid_yield():
    """Test error when sweep yields non-dict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir).resolve()
        script_path = tmpdir / "train.py"
        script_path.write_text("print('train')")
        sweep_file = create_sweep_file(tmpdir)

        local_obj = LocalSweep(
            script_path=str(script_path),
            sweep_file=str(sweep_file),
            sweep_function="invalid_sweep",
        )

        with pytest.raises(TypeError, match="must yield dictionaries"):
            local_obj.generate_param_sets()


def test_local_sweep_gpu_assignment():
    """Test GPU assignment to workers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir).resolve()
        script_path = tmpdir / "train.py"
        script_path.write_text("print('train')")
        sweep_file = create_sweep_file(tmpdir)

        local_obj = LocalSweep(
            script_path=str(script_path),
            sweep_file=str(sweep_file),
            sweep_function="grid_sweep",
            workers=4,
            gpus="0,1,2,3",
        )

        # Each worker should get one GPU
        assert local_obj.get_gpu_for_worker(0) == "0"
        assert local_obj.get_gpu_for_worker(1) == "1"
        assert local_obj.get_gpu_for_worker(2) == "2"
        assert local_obj.get_gpu_for_worker(3) == "3"


def test_local_sweep_gpu_assignment_multiple_per_worker():
    """Test GPU assignment with multiple GPUs per worker."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir).resolve()
        script_path = tmpdir / "train.py"
        script_path.write_text("print('train')")
        sweep_file = create_sweep_file(tmpdir)

        local_obj = LocalSweep(
            script_path=str(script_path),
            sweep_file=str(sweep_file),
            sweep_function="grid_sweep",
            workers=2,
            gpus="0,1,2,3",
        )

        # Each worker should get two GPUs
        assert local_obj.get_gpu_for_worker(0) == "0,1"
        assert local_obj.get_gpu_for_worker(1) == "2,3"


def test_local_sweep_job_distribution():
    """Test job distribution across workers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir).resolve()
        script_path = tmpdir / "train.py"
        script_path.write_text("print('train')")
        sweep_file = create_sweep_file(tmpdir)

        local_obj = LocalSweep(
            script_path=str(script_path),
            sweep_file=str(sweep_file),
            sweep_function="grid_sweep",  # 4 jobs
            workers=3,
        )

        local_obj.generate_param_sets()
        distribution = local_obj.distribute_jobs()

        # 4 jobs across 3 workers: [0, 3], [1], [2]
        assert len(distribution) == 3
        assert 0 in distribution[0]
        assert 3 in distribution[0]
        assert 1 in distribution[1]
        assert 2 in distribution[2]


def test_local_sweep_save_scripts():
    """Test saving worker scripts and mapping."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir).resolve()
        script_path = tmpdir / "train.py"
        script_path.write_text("print('train')")
        sweep_file = create_sweep_file(tmpdir)

        # Create .smanager
        smanager_dir = tmpdir / ".smanager"
        smanager_dir.mkdir()

        local_obj = LocalSweep(
            script_path=str(script_path),
            sweep_file=str(sweep_file),
            sweep_function="grid_sweep",
            workers=2,
            gpus="0,1",
        )

        paths = local_obj.save_scripts()

        assert len(paths) == 2
        assert all(p.exists() for p in paths)
        assert all(p.suffix == ".sh" for p in paths)

        # Check sweep.json was created
        sweep_json = local_obj.sweep_dir / "sweep.json"
        assert sweep_json.exists()

        # Check sweep.json content
        with open(sweep_json, encoding="utf-8") as f:
            data = json.load(f)
        assert data["type"] == "local"
        assert data["workers"] == 2
        assert data["total_jobs"] == 4

        # Check logs directory was created
        logs_dir = local_obj.sweep_dir / "logs"
        assert logs_dir.exists()


def test_local_sweep_script_contains_preamble():
    """Test that generated scripts contain the preamble."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir).resolve()
        script_path = tmpdir / "train.py"
        script_path.write_text("print('train')")
        sweep_file = create_sweep_file(tmpdir)

        # Create .smanager with preamble
        smanager_dir = tmpdir / ".smanager"
        smanager_dir.mkdir()
        create_preamble(smanager_dir)

        local_obj = LocalSweep(
            script_path=str(script_path),
            sweep_file=str(sweep_file),
            sweep_function="simple_sweep",
            workers=1,
        )

        paths = local_obj.save_scripts()

        content = paths[0].read_text()
        assert "source ~/miniconda3/bin/activate" in content
        assert "conda activate myenv" in content


def test_local_sweep_script_contains_cuda_visible_devices():
    """Test that generated scripts set CUDA_VISIBLE_DEVICES."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir).resolve()
        script_path = tmpdir / "train.py"
        script_path.write_text("print('train')")
        sweep_file = create_sweep_file(tmpdir)

        # Create .smanager
        smanager_dir = tmpdir / ".smanager"
        smanager_dir.mkdir()

        local_obj = LocalSweep(
            script_path=str(script_path),
            sweep_file=str(sweep_file),
            sweep_function="simple_sweep",
            workers=2,
            gpus="0,1",
        )

        paths = local_obj.save_scripts()

        # Worker 0 should have GPU 0
        content0 = paths[0].read_text()
        assert "export CUDA_VISIBLE_DEVICES=0" in content0

        # Worker 1 should have GPU 1
        content1 = paths[1].read_text()
        assert "export CUDA_VISIBLE_DEVICES=1" in content1


def test_local_sweep_len():
    """Test __len__ method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir).resolve()
        script_path = tmpdir / "train.py"
        script_path.write_text("print('train')")
        sweep_file = create_sweep_file(tmpdir)

        local_obj = LocalSweep(
            script_path=str(script_path),
            sweep_file=str(sweep_file),
            sweep_function="grid_sweep",
        )

        assert len(local_obj) == 4


def test_local_sweep_custom_arg_format():
    """Test custom argument format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir).resolve()
        script_path = tmpdir / "train.py"
        script_path.write_text("print('train')")
        sweep_file = create_sweep_file(tmpdir)

        # Create .smanager
        smanager_dir = tmpdir / ".smanager"
        smanager_dir.mkdir()

        local_obj = LocalSweep(
            script_path=str(script_path),
            sweep_file=str(sweep_file),
            sweep_function="simple_sweep",
            arg_format="{key}={value}",  # No -- prefix
            workers=1,
        )

        paths = local_obj.save_scripts()

        content = paths[0].read_text()
        assert "lr=0.1" in content
        assert "lr=0.01" in content


@patch("smanager.local.subprocess.run")
def test_local_sweep_list_sessions(mock_run):
    """Test listing tmux sessions."""
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = "sweep_0\nsweep_1\nother_session\n"

    sessions = LocalSweep.list_sweep_sessions("sweep")

    assert sessions == ["sweep_0", "sweep_1"]
    mock_run.assert_called_once()


@patch("smanager.local.subprocess.run")
def test_local_sweep_list_sessions_no_tmux(mock_run):
    """Test listing sessions when tmux not running."""
    mock_run.return_value.returncode = 1

    sessions = LocalSweep.list_sweep_sessions("sweep")

    assert sessions == []


def test_local_sweep_escapes_special_chars_in_echo():
    """Test that special characters in params are escaped for bash echo."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir).resolve()
        script_path = tmpdir / "train.py"
        script_path.write_text("print('train')")

        # Create sweep with special characters (parentheses, quotes, etc.)
        sweep_file = tmpdir / "special_sweeps.py"
        sweep_file.write_text(
            '''
def special_sweep():
    """Sweep with special bash characters."""
    yield {"shape": '"(4,4,1)"', "cmd": "$HOME", "backtick": "`echo hi`"}
'''
        )

        # Create .smanager
        smanager_dir = tmpdir / ".smanager"
        smanager_dir.mkdir()

        local_obj = LocalSweep(
            script_path=str(script_path),
            sweep_file=str(sweep_file),
            sweep_function="special_sweep",
            workers=1,
        )

        paths = local_obj.save_scripts()
        content = paths[0].read_text()

        # Verify that special characters are properly escaped in echo statements
        # Double quotes should be escaped as \"
        assert r"\"(4,4,1)\"" in content
        # Dollar sign should be escaped as \$
        assert r"\$HOME" in content
        # Backticks should be escaped as \`
        assert r"\`echo hi\`" in content

        # The script should be valid bash - no syntax errors
        # We can verify by checking bash -n (syntax check only)
        import subprocess  # pylint: disable=import-outside-toplevel

        result = subprocess.run(
            ["bash", "-n", str(paths[0])],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"Bash syntax error: {result.stderr}"
