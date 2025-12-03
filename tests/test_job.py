"""Tests for job creation and handling."""

import tempfile
from pathlib import Path

from smanager.job import SlurmJob


def test_job_uuid_generation():
    """Test that jobs get unique UUIDs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir).resolve()
        script_path = tmpdir / "script.py"
        script_path.write_text("print('hello')")

        job1 = SlurmJob(script_path=str(script_path))
        job2 = SlurmJob(script_path=str(script_path))

        assert job1.job_uuid != job2.job_uuid
        assert len(job1.job_uuid) == 36  # UUID format


def test_job_script_generation():
    """Test sbatch script generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir).resolve()
        script_path = tmpdir / "train.py"
        script_path.write_text("print('training')")

        job = SlurmJob(
            script_path=str(script_path),
            gpus=4,
            memory="32G",
            time="24:00:00",
            partition="gpu",
        )

        script = job.generate_script()

        assert "#!/bin/bash" in script
        assert f"--job-name={job.job_uuid}" in script
        assert "--gres=gpu:4" in script
        assert "--mem=32G" in script
        assert "--time=24:00:00" in script
        assert "--partition=gpu" in script
        # Use resolved path for comparison
        assert str(script_path.resolve()) in script


def test_job_with_script_args():
    """Test job with script arguments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir).resolve()
        script_path = tmpdir / "train.py"
        script_path.write_text("import sys; print(sys.argv)")

        job = SlurmJob(
            script_path=str(script_path),
            script_args=["--lr", "0.01", "--epochs", "100"],
        )

        script = job.generate_script()

        assert "--lr 0.01 --epochs 100" in script


def test_job_save_script():
    """Test saving job script to disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir).resolve()
        script_path = tmpdir / "train.py"
        script_path.write_text("print('train')")

        # Create .smanager directory
        smanager_dir = tmpdir / ".smanager"
        smanager_dir.mkdir()

        job = SlurmJob(script_path=str(script_path), gpus=2)
        saved_path = job.save_script()

        assert saved_path.exists()
        assert saved_path.suffix == ".sbatch"
        assert job.job_uuid in saved_path.name

        # Check logs directory was created
        logs_dir = saved_path.parent / "logs"
        assert logs_dir.exists()


def test_job_experiment_name_default():
    """Test that experiment name defaults to parent.script format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir).resolve()
        # Create nested structure: experiments/train.py
        exp_dir = tmpdir / "experiments"
        exp_dir.mkdir()
        script_path = exp_dir / "train.py"
        script_path.write_text("print('train')")

        job = SlurmJob(script_path=str(script_path))

        assert job.experiment_name == "experiments.train"


def test_job_experiment_name_custom():
    """Test custom experiment name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir).resolve()
        script_path = tmpdir / "train.py"
        script_path.write_text("print('train')")

        job = SlurmJob(
            script_path=str(script_path),
            experiment_name="my_experiment",
        )

        assert job.experiment_name == "my_experiment"
