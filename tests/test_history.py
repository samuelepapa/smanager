"""Tests for job history discovery and Slurm status helpers."""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import patch

from smanager.history import discover_jobs, refresh_status


def _write_single_job_manifest(
    job_dir: Path,
    *,
    job_uuid: str,
    slurm_job_id: Optional[str],
    dry_run: bool,
    created_at: str,
    submitted_at: Optional[str],
) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / f"{job_uuid}.sbatch").write_text(
        "#!/bin/bash\necho test\n", encoding="utf-8"
    )
    (job_dir / "logs").mkdir(exist_ok=True)
    (job_dir / f"{job_uuid}.out").write_text("stdout", encoding="utf-8")
    (job_dir / f"{job_uuid}.err").write_text("stderr", encoding="utf-8")
    manifest = {
        "kind": "job",
        "job_uuid": job_uuid,
        "created_at": created_at,
        "submitted_at": submitted_at,
        "dry_run": dry_run,
        "experiment_name": "demo",
        "script_path": str(job_dir.parent.parent / "train.py"),
        "script_args": ["--lr", "0.1"],
        "sbatch_path": str(job_dir / f"{job_uuid}.sbatch"),
        "output": str(job_dir / f"{job_uuid}.out"),
        "error": str(job_dir / f"{job_uuid}.err"),
        "slurm_job_id": slurm_job_id,
        "slurm_options": {"partition": "gpu", "gpus": 2},
        "hyperparameters": {},
    }
    (job_dir / f"{job_uuid}.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def _write_sweep_manifest(
    sweep_dir: Path,
    *,
    job_uuid: str,
    slurm_job_id: Optional[str],
    dry_run: bool,
    created_at: str,
    submitted_at: Optional[str],
) -> None:
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / f"{job_uuid}.sbatch").write_text(
        "#!/bin/bash\necho sweep\n", encoding="utf-8"
    )
    (sweep_dir / "logs").mkdir(exist_ok=True)
    (sweep_dir / f"{job_uuid}.out").write_text("sweep stdout", encoding="utf-8")
    (sweep_dir / f"{job_uuid}.err").write_text("sweep stderr", encoding="utf-8")
    sweep_manifest = {
        "kind": "sweep",
        "sweep_uuid": "20260324120000.abcd1234",
        "created_at": created_at,
        "dry_run": dry_run,
        "script": str(sweep_dir.parent.parent.parent / "train.py"),
        "script_path": str(sweep_dir.parent.parent.parent / "train.py"),
        "sweep_file": str(sweep_dir.parent.parent.parent / "sweeps.py"),
        "sweep_function": "grid",
        "base_args": [],
        "arg_format": "--{key}={value}",
        "total_jobs": 1,
        "jobs": {
            job_uuid: {
                "job_uuid": job_uuid,
                "index": 0,
                "params": {"lr": 0.1},
                "slurm_job_id": slurm_job_id,
                "created_at": created_at,
                "submitted_at": submitted_at,
                "dry_run": dry_run,
                "script_args": ["--lr=0.1"],
                "sbatch_path": str(sweep_dir / f"{job_uuid}.sbatch"),
                "output": str(sweep_dir / f"{job_uuid}.out"),
                "error": str(sweep_dir / f"{job_uuid}.err"),
                "slurm_options": {"partition": "gpu", "gpus": 4},
            },
        },
    }
    (sweep_dir / "sweep.json").write_text(
        json.dumps(sweep_manifest, indent=2),
        encoding="utf-8",
    )


def test_discover_jobs_filters_dry_run():
    """Test discovery across job and sweep manifests with dry-run filtering."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()
        script_dir = root / ".smanager" / "scripts"

        _write_single_job_manifest(
            script_dir / "demo" / "run1",
            job_uuid="20260324120000.aaaa1111",
            slurm_job_id="12345",
            dry_run=False,
            created_at="2026-03-24T12:00:00",
            submitted_at="2026-03-24T12:01:00",
        )
        _write_single_job_manifest(
            script_dir / "demo" / "run2",
            job_uuid="20260324120000.bbbb2222",
            slurm_job_id=None,
            dry_run=True,
            created_at="2026-03-24T12:02:00",
            submitted_at=None,
        )
        _write_sweep_manifest(
            script_dir / "demo" / "grid" / "20260324120000.cddd3333",
            job_uuid="20260324120000.cddd3333",
            slurm_job_id="54321",
            dry_run=False,
            created_at="2026-03-24T12:03:00",
            submitted_at="2026-03-24T12:04:00",
        )

        visible = discover_jobs(script_dir, include_dry_run=False)
        all_jobs = discover_jobs(script_dir, include_dry_run=True)

        assert len(visible) == 2
        assert len(all_jobs) == 3
        assert all(not record.dry_run for record in visible)
        assert any(record.job_uuid == "20260324120000.cddd3333" for record in visible)


def test_refresh_status_maps_running_and_done_jobs():
    """Test that refresh_status reads squeue first and sacct for finished jobs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()
        script_dir = root / ".smanager" / "scripts"

        _write_single_job_manifest(
            script_dir / "demo" / "run1",
            job_uuid="20260324120000.aaaa1111",
            slurm_job_id="12345",
            dry_run=False,
            created_at="2026-03-24T12:00:00",
            submitted_at="2026-03-24T12:01:00",
        )
        _write_sweep_manifest(
            script_dir / "demo" / "grid" / "20260324120000.cddd3333",
            job_uuid="20260324120000.cddd3333",
            slurm_job_id="54321",
            dry_run=False,
            created_at="2026-03-24T12:03:00",
            submitted_at="2026-03-24T12:04:00",
        )

        records = discover_jobs(script_dir, include_dry_run=False)

        def fake_run(cmd, **_kwargs):
            if cmd[0] == "squeue":
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout="12345|R|00:10:00|node01|job\n",
                    stderr="",
                )
            if cmd[0] == "sacct":
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout="54321|CD|00:20:00|node02|0:0\n",
                    stderr="",
                )
            raise AssertionError(f"Unexpected command: {cmd}")

        with patch("smanager.history.subprocess.run", side_effect=fake_run):
            refresh_status(records)

        record_by_id = {record.slurm_job_id: record for record in records}
        assert record_by_id["12345"].status_category == "running"
        assert record_by_id["12345"].status_state == "R"
        assert record_by_id["54321"].status_category == "done"
        assert record_by_id["54321"].status_state == "CD"
