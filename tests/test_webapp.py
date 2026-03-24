"""Integration tests for the SManager web dashboard."""

from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import patch

from smanager.webapp import create_app


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_job_manifest(
    job_dir: Path,
    *,
    job_uuid: str,
    experiment_name: str,
    script_path: Path,
    dry_run: bool,
    slurm_job_id: Optional[str],
    stdout: str,
    stderr: str,
    sbatch: str,
    script_args: Optional[List[str]] = None,
    hyperparameters: Optional[Dict[str, object]] = None,
    slurm_options: Optional[Dict[str, object]] = None,
) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)

    sbatch_path = job_dir / f"{job_uuid}.sbatch"
    output_path = job_dir / f"{job_uuid}.out"
    error_path = job_dir / f"{job_uuid}.err"
    _write_text(sbatch_path, sbatch)
    _write_text(output_path, stdout)
    _write_text(error_path, stderr)

    manifest = {
        "kind": "job",
        "job_uuid": job_uuid,
        "created_at": "2026-03-24T12:00:00",
        "submitted_at": None if dry_run else "2026-03-24T12:01:00",
        "dry_run": dry_run,
        "experiment_name": experiment_name,
        "script_path": str(script_path),
        "script_args": script_args or ["--lr", "0.1"],
        "sbatch_path": str(sbatch_path),
        "output": str(output_path),
        "error": str(error_path),
        "slurm_job_id": slurm_job_id,
        "slurm_options": slurm_options or {"partition": "gpu", "gpus": 2},
        "hyperparameters": hyperparameters or {"lr": 0.1},
    }
    _write_text(job_dir / f"{job_uuid}.json", json.dumps(manifest, indent=2))


def _write_sweep_manifest(
    sweep_dir: Path,
    *,
    sweep_uuid: str,
    experiment_name: str,
    script_path: Path,
    sweep_file: Path,
    job_specs: Dict[str, Dict[str, object]],
) -> None:
    sweep_dir.mkdir(parents=True, exist_ok=True)

    jobs: Dict[str, Dict[str, object]] = {}
    for job_uuid, spec in job_specs.items():
        sbatch_path = sweep_dir / f"{job_uuid}.sbatch"
        output_path = sweep_dir / f"{job_uuid}.out"
        error_path = sweep_dir / f"{job_uuid}.err"
        _write_text(sbatch_path, str(spec["sbatch"]))
        _write_text(output_path, str(spec["stdout"]))
        _write_text(error_path, str(spec["stderr"]))
        jobs[job_uuid] = {
            "job_uuid": job_uuid,
            "index": spec["index"],
            "params": spec["params"],
            "slurm_job_id": spec["slurm_job_id"],
            "created_at": "2026-03-24T12:03:00",
            "submitted_at": spec["submitted_at"],
            "dry_run": spec["dry_run"],
            "script_args": spec["script_args"],
            "sbatch_path": str(sbatch_path),
            "output": str(output_path),
            "error": str(error_path),
            "slurm_options": spec["slurm_options"],
        }

    sweep_manifest = {
        "kind": "sweep",
        "sweep_uuid": sweep_uuid,
        "created_at": "2026-03-24T12:02:00",
        "dry_run": False,
        "experiment_name": experiment_name,
        "script_path": str(script_path),
        "sweep_file": str(sweep_file),
        "sweep_function": "grid_search",
        "base_args": ["--epochs", "50"],
        "arg_format": "--{key}={value}",
        "total_jobs": len(jobs),
        "jobs": jobs,
    }
    _write_text(sweep_dir / "sweep.json", json.dumps(sweep_manifest, indent=2))


def _build_demo_project(root: Path) -> Dict[str, str]:
    script_dir = root / ".smanager" / "scripts"
    _write_text(
        root / ".smanager" / "config.yaml",
        "partition: gpu\ngpus: 2\nmemory: 16G\n",
    )
    _write_text(
        root / ".smanager" / "preamble.sh",
        "# Fake preamble for dashboard smoke tests\nexport DEMO_RUN=1\n",
    )
    _write_text(
        root / "train.py",
        "import sys\nprint('demo train', sys.argv[1:])\n",
    )
    _write_text(
        root / "sweeps.py",
        "def grid_search():\n"
        "    yield {'lr': 0.1, 'batch_size': 32}\n"
        "    yield {'lr': 0.01, 'batch_size': 64}\n",
    )

    single_job_uuid = "20260324120000.aaaa1111"
    dry_run_job_uuid = "20260324120000.bbbb2222"
    sweep_uuid = "20260324120000.cddd3333"
    sweep_done_uuid = "20260324120000.dddd4444"
    sweep_failed_uuid = "20260324120000.eeee5555"

    _write_job_manifest(
        script_dir / "demo" / "single" / single_job_uuid,
        job_uuid=single_job_uuid,
        experiment_name="demo.single",
        script_path=root / "train.py",
        dry_run=False,
        slurm_job_id="12345",
        stdout="single stdout line\n",
        stderr="single stderr line\n",
        sbatch="#!/bin/bash\necho single\n",
        script_args=["--lr", "0.1", "--epochs", "50"],
        hyperparameters={"lr": 0.1, "epochs": 50},
        slurm_options={"partition": "gpu", "gpus": 2, "memory": "16G"},
    )
    _write_job_manifest(
        script_dir / "demo" / "single" / dry_run_job_uuid,
        job_uuid=dry_run_job_uuid,
        experiment_name="demo.single",
        script_path=root / "train.py",
        dry_run=True,
        slurm_job_id=None,
        stdout="dry run stdout\n",
        stderr="dry run stderr\n",
        sbatch="#!/bin/bash\necho dry run\n",
        script_args=["--lr", "0.2"],
        hyperparameters={"lr": 0.2},
        slurm_options={"partition": "gpu", "gpus": 1},
    )
    _write_sweep_manifest(
        script_dir / "demo" / "sweep" / sweep_uuid,
        sweep_uuid=sweep_uuid,
        experiment_name="demo.sweep",
        script_path=root / "train.py",
        sweep_file=root / "sweeps.py",
        job_specs={
            sweep_done_uuid: {
                "index": 0,
                "params": {"lr": 0.1, "batch_size": 32},
                "slurm_job_id": "54321",
                "submitted_at": "2026-03-24T12:04:00",
                "dry_run": False,
                "script_args": ["--epochs", "50", "--lr=0.1", "--batch_size=32"],
                "sbatch": "#!/bin/bash\necho sweep done\n",
                "stdout": "sweep stdout done\n",
                "stderr": "sweep stderr done\n",
                "slurm_options": {"partition": "gpu", "gpus": 4},
            },
            sweep_failed_uuid: {
                "index": 1,
                "params": {"lr": 0.01, "batch_size": 64},
                "slurm_job_id": "67890",
                "submitted_at": "2026-03-24T12:05:00",
                "dry_run": False,
                "script_args": ["--epochs", "50", "--lr=0.01", "--batch_size=64"],
                "sbatch": "#!/bin/bash\necho sweep failed\n",
                "stdout": "sweep stdout failed\n",
                "stderr": "sweep stderr failed\n",
                "slurm_options": {"partition": "gpu", "gpus": 4},
            },
        },
    )

    return {
        "single_job_uuid": single_job_uuid,
        "dry_run_job_uuid": dry_run_job_uuid,
        "sweep_done_uuid": sweep_done_uuid,
        "sweep_failed_uuid": sweep_failed_uuid,
        "sweep_uuid": sweep_uuid,
    }


def _fake_refresh_status(status_map: Dict[str, Dict[str, str]]):
    def _refresh(records):
        for record in records:
            status = status_map.get(record.job_uuid)
            if status is None:
                continue
            record.status_state = status["state"]
            record.status_time = status["time"]
            record.status_node = status["node"]
            record.status_exit_code = status["exit_code"]
        return records

    return _refresh


@contextmanager
def _demo_client():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()
        ids = _build_demo_project(root)
        app = create_app(project_root=root)
        client = app.test_client()
        status_map = {
            ids["single_job_uuid"]: {
                "state": "R",
                "time": "00:10:00",
                "node": "node01",
                "exit_code": "-",
            },
            ids["sweep_done_uuid"]: {
                "state": "CD",
                "time": "00:20:00",
                "node": "node02",
                "exit_code": "0:0",
            },
            ids["sweep_failed_uuid"]: {
                "state": "F",
                "time": "00:03:00",
                "node": "node03",
                "exit_code": "1:0",
            },
        }

        yield root, ids, client, status_map


def test_dashboard_renders_jobs_and_dry_run_toggle():
    """Exercise the dashboard and dry-run toggle."""
    with _demo_client() as (_root, ids, client, status_map):
        with patch(
            "smanager.webapp.refresh_status",
            side_effect=_fake_refresh_status(status_map),
        ):
            dashboard = client.get("/")
            assert dashboard.status_code == 200
            body = dashboard.get_data(as_text=True)
            assert ids["single_job_uuid"] in body
            assert ids["sweep_done_uuid"] in body
            assert ids["sweep_failed_uuid"] in body
            assert ids["dry_run_job_uuid"] not in body
            assert "running (R)" in body
            assert "done (CD)" in body
            assert "failed (F)" in body

            dashboard_all = client.get("/?show_dry_run=1")
            assert dashboard_all.status_code == 200
            body_all = dashboard_all.get_data(as_text=True)
            assert ids["dry_run_job_uuid"] in body_all
            assert "dry-run" in body_all


def test_job_detail_tabs_status_and_kill_flow():
    """Exercise job detail tabs, status fragment, and kill flow."""
    with _demo_client() as (_root, ids, client, status_map):
        with patch(
            "smanager.webapp.refresh_status",
            side_effect=_fake_refresh_status(status_map),
        ):
            detail = client.get(f"/jobs/{ids['single_job_uuid']}")
            assert detail.status_code == 200
            detail_body = detail.get_data(as_text=True)
            assert "Job Detail" in detail_body
            assert "stdout" in detail_body
            assert "stderr" in detail_body
            assert "hyperparameters" in detail_body
            assert ".sbatch" in detail_body
            assert "single stdout line" in detail_body
            assert '<button type="submit">Kill job</button>' not in detail_body
            assert '<button class="btn" type="submit">Kill job</button>' in detail_body
            assert '<button type="button" class="tab-button active"' in detail_body

            stderr_detail = client.get(f"/jobs/{ids['single_job_uuid']}?tab=stderr")
            stderr_detail_body = stderr_detail.get_data(as_text=True)
            assert (
                '<button type="button" class="tab-button active"' in stderr_detail_body
            )
            assert (
                'hx-get="/jobs/20260324120000.aaaa1111/tab/stderr"'
                in stderr_detail_body
            )

            stdout_tab = client.get(f"/jobs/{ids['single_job_uuid']}/tab/stdout")
            stderr_tab = client.get(f"/jobs/{ids['single_job_uuid']}/tab/stderr")
            hyper_tab = client.get(
                f"/jobs/{ids['single_job_uuid']}/tab/hyperparameters"
            )
            sbatch_tab = client.get(f"/jobs/{ids['single_job_uuid']}/tab/sbatch")

            assert "single stdout line" in stdout_tab.get_data(as_text=True)
            assert "single stderr line" in stderr_tab.get_data(as_text=True)
            assert "script_args" in hyper_tab.get_data(as_text=True)
            assert "lr" in hyper_tab.get_data(as_text=True)
            assert "#!/bin/bash" in sbatch_tab.get_data(as_text=True)

            status_fragment = client.get(f"/jobs/{ids['single_job_uuid']}/status")
            status_body = status_fragment.get_data(as_text=True)
            assert "Slurm State" in status_body
            assert "R" in status_body
            assert "node01" in status_body

            with patch("smanager.webapp.cancel_job") as cancel_mock:
                kill_response = client.post(
                    f"/jobs/{ids['single_job_uuid']}/kill",
                    follow_redirects=False,
                )
                assert kill_response.status_code in {302, 303}
                cancel_mock.assert_called_once_with("12345")

            dry_detail = client.get(f"/jobs/{ids['dry_run_job_uuid']}")
            assert dry_detail.status_code == 200
            assert "Kill job</button>" not in dry_detail.get_data(as_text=True)


def test_kill_job_command_error_is_rendered():
    """If the local kill command is missing, show the error in the page."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()
        ids = _build_demo_project(root)
        app = create_app(project_root=root)
        client = app.test_client()

        with patch(
            "smanager.webapp.cancel_job",
            side_effect=FileNotFoundError(2, "No such file or directory", "scancel"),
        ):
            response = client.post(
                f"/jobs/{ids['single_job_uuid']}/kill?tab=stderr",
                follow_redirects=False,
            )

        assert response.status_code == 200
        body = response.get_data(as_text=True)
        assert "Kill job failed:" in body
        assert "Failed to kill job 12345" in body
        assert "scancel" in body
        assert 'hx-get="/jobs/20260324120000.aaaa1111/tab/stderr"' in body


def test_dashboard_404_for_missing_job():
    """Missing jobs should return a 404 instead of rendering a broken page."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()
        _build_demo_project(root)
        app = create_app(project_root=root)
        client = app.test_client()

        with patch(
            "smanager.webapp.refresh_status", side_effect=lambda records: records
        ):
            response = client.get("/jobs/does-not-exist")

        assert response.status_code == 404
