"""Integration tests for the SManager web dashboard."""

from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

from smanager.webapp import create_app, serve


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


def _write_local_sweep_manifest(
    sweep_dir: Path,
    *,
    sweep_uuid: str,
    script_path: Path,
    sweep_file: Path,
) -> None:
    """Write the worker-oriented manifest produced by LocalSweep."""
    worker_path = sweep_dir / "worker_0.sh"
    log_path = sweep_dir / "logs" / "worker_0.log"
    _write_text(worker_path, "#!/bin/bash\necho local sweep\n")
    _write_text(log_path, "local sweep output\n")
    manifest = {
        "sweep_uuid": sweep_uuid,
        "type": "local",
        "experiment_name": "demo.local",
        "script": str(script_path),
        "sweep_file": str(sweep_file),
        "sweep_function": "local_grid",
        "base_args": ["--epochs", "2"],
        "arg_format": "--{key}={value}",
        "workers": 1,
        "gpus": "0",
        "total_jobs": 2,
        "created_at": "2026-03-24T12:06:00",
        "jobs": {
            "0": {"index": 0, "params": {"lr": 0.1}},
            "1": {"index": 1, "params": {"lr": 0.01}},
        },
        "worker_assignments": {
            "worker_0": {
                "job_indices": [0, 1],
                "gpu": "0",
                "session_name": "sweep_0",
            }
        },
    }
    _write_text(sweep_dir / "sweep.json", json.dumps(manifest, indent=2))


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
    local_sweep_uuid = "20260324120000.ffff5555"

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
    _write_local_sweep_manifest(
        script_dir / "demo" / "local" / local_sweep_uuid,
        sweep_uuid=local_sweep_uuid,
        script_path=root / "train.py",
        sweep_file=root / "sweeps.py",
    )

    return {
        "single_job_uuid": single_job_uuid,
        "dry_run_job_uuid": dry_run_job_uuid,
        "sweep_done_uuid": sweep_done_uuid,
        "sweep_failed_uuid": sweep_failed_uuid,
        "sweep_uuid": sweep_uuid,
        "local_sweep_uuid": local_sweep_uuid,
        "local_job_uuid": f"{local_sweep_uuid}.local-0",
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
            assert f"/sweeps/{ids['local_sweep_uuid']}" in body
            assert "local sweep output" not in body
            assert ids["dry_run_job_uuid"] not in body
            assert f"/sweeps/{ids['sweep_uuid']}" in body
            assert "status-bar" in body
            assert "running (R)" in body
            assert "done (CD)" in body
            assert "failed (F)" in body

            dashboard_all = client.get("/?show_dry_run=1")
            assert dashboard_all.status_code == 200
            body_all = dashboard_all.get_data(as_text=True)
            assert ids["dry_run_job_uuid"] in body_all
            assert "dry-run" in body_all


def test_dashboard_columns_default_menu_and_actions():
    """Dashboard defaults to the compact column set and bulk actions only."""
    with _demo_client() as (_root, ids, client, status_map):
        with patch(
            "smanager.webapp.refresh_status",
            side_effect=_fake_refresh_status(status_map),
        ):
            response = client.get("/")

        assert response.status_code == 200
        body = response.get_data(as_text=True)
        assert ids["single_job_uuid"] in body
        assert 'id="columns-button"' in body
        assert 'class="bulk-actions hidden"' in body
        assert "Actions" not in body
        assert 'data-confirm="Delete this job?"' not in body

        assert 'class="dashboard-column column-job"' in body
        assert 'data-column="job">Job</th>' in body
        assert 'class="dashboard-column column-run-time"' in body
        assert 'data-column="run_time">Run Time</th>' in body
        assert 'class="dashboard-column column-duration"' in body
        assert 'data-column="duration">Duration</th>' in body
        assert 'class="dashboard-column column-partition"' in body
        assert 'data-column="partition">Partition</th>' in body
        assert 'class="dashboard-column column-status"' in body
        assert 'data-column="status">Status</th>' in body
        assert '<th class="dashboard-column column-gpus hidden"' in body
        assert '<th class="dashboard-column column-slurm-id hidden"' in body
        assert '<th class="dashboard-column column-type hidden"' in body

        for column_key in (
            "job",
            "run_time",
            "duration",
            "gpus",
            "partition",
            "status",
            "slurm_id",
            "type",
        ):
            assert f'data-column="{column_key}"' in body


def test_dashboard_columns_persist_in_project_settings():
    """Column choices are normalized, saved, and used on later requests."""
    with _demo_client() as (root, _ids, client, status_map):
        with patch(
            "smanager.webapp.refresh_status",
            side_effect=_fake_refresh_status(status_map),
        ):
            response = client.post(
                "/settings/dashboard-columns",
                json={"dashboard_columns": ["type", "job", "unknown", "type"]},
            )

        assert response.status_code == 200
        assert response.get_json() == {"dashboard_columns": ["job", "type"]}

        settings_path = root / ".smanager" / "webapp_settings.json"
        assert json.loads(settings_path.read_text(encoding="utf-8")) == {
            "dashboard_columns": ["job", "type"]
        }

        with patch(
            "smanager.webapp.refresh_status",
            side_effect=_fake_refresh_status(status_map),
        ):
            dashboard = client.get("/")
        body = dashboard.get_data(as_text=True)
        assert 'class="dashboard-column column-job"' in body
        assert 'data-column="job">Job</th>' in body
        assert 'class="dashboard-column column-type"' in body
        assert 'data-column="type">Type</th>' in body
        assert '<th class="dashboard-column column-status hidden"' in body


def test_dashboard_columns_settings_preserve_other_values_and_recover_invalid_data():
    """Invalid settings use defaults and saving retains unrelated preferences."""
    with _demo_client() as (root, _ids, client, status_map):
        settings_path = root / ".smanager" / "webapp_settings.json"
        for invalid_settings in (
            '{"theme": "dark", "dashboard_columns": "not-a-list"}',
            '{"theme": "dark", "dashboard_columns": ["unknown"]}',
            '{"theme": "dark", "dashboard_columns": [[], "job"]}',
            "not valid json",
        ):
            settings_path.write_text(invalid_settings, encoding="utf-8")
            with patch(
                "smanager.webapp.refresh_status",
                side_effect=_fake_refresh_status(status_map),
            ):
                dashboard = client.get("/")
            body = dashboard.get_data(as_text=True)
            assert 'class="dashboard-column column-job"' in body
            assert 'data-column="job">Job</th>' in body
            assert 'class="dashboard-column column-gpus hidden"' in body

        settings_path.write_text(
            json.dumps({"theme": "dark", "dashboard_columns": ["unknown"]}),
            encoding="utf-8",
        )
        response = client.post(
            "/settings/dashboard-columns",
            json={"dashboard_columns": []},
        )
        assert response.status_code == 200
        assert response.get_json() == {
            "dashboard_columns": ["job", "run_time", "duration", "partition", "status"]
        }
        assert json.loads(settings_path.read_text(encoding="utf-8"))["theme"] == "dark"


def test_dashboard_empty_state_uses_fixed_column_count():
    """The empty state spans the selector and all supported data columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()
        (root / ".smanager").mkdir()
        (root / ".smanager" / "config.yaml").write_text("{}\n", encoding="utf-8")
        app = create_app(project_root=root)
        client = app.test_client()

        with patch(
            "smanager.webapp.refresh_status",
            side_effect=lambda records: records,
        ):
            response = client.get("/")

        assert response.status_code == 200
        body = response.get_data(as_text=True)
        assert 'colspan="9"' in body
        assert 'class="empty-state">No jobs found' in body


def test_sweep_detail_renders_jobs_and_parameter_tabs():
    """Sweep detail page shows all jobs plus parameter-specific tabs."""
    with _demo_client() as (_root, ids, client, status_map):
        with patch(
            "smanager.webapp.refresh_status",
            side_effect=_fake_refresh_status(status_map),
        ):
            response = client.get(f"/sweeps/{ids['sweep_uuid']}")
            assert response.status_code == 200
            body = response.get_data(as_text=True)
            assert ids["sweep_done_uuid"] in body
            assert ids["sweep_failed_uuid"] in body
            assert 'data-param-tab="lr"' in body
            assert 'data-param-tab="batch_size"' in body

            local_response = client.get(f"/sweeps/{ids['local_sweep_uuid']}")
            assert local_response.status_code == 200
            local_body = local_response.get_data(as_text=True)
            assert ids["local_job_uuid"] in local_body
            assert 'data-param-tab="lr"' in local_body

            local_job = client.get(f"/jobs/{ids['local_job_uuid']}")
            assert local_job.status_code == 200
            assert "local sweep output" in local_job.get_data(as_text=True)


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
            status_position = detail_body.index(
                '<div id="status-panel" class="card status-card"'
            )
            overview_position = detail_body.index(
                '<div class="card job-overview-card">'
            )
            assert status_position < overview_position
            assert "stdout" in detail_body
            assert "stderr" in detail_body
            assert "hyperparameters" in detail_body
            assert ".sbatch" in detail_body
            assert "single stdout line" in detail_body
            assert '<button type="submit">Kill job</button>' not in detail_body
            assert 'data-confirm="Kill this job?"' in detail_body
            assert 'data-confirm="Delete this job?"' in detail_body
            assert "Copy .sbatch path" in detail_body
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


def test_dashboard_selection_and_delete_actions():
    """Dashboard exposes single-row and bulk delete/kill controls."""
    with _demo_client() as (_root, ids, client, status_map):
        with patch(
            "smanager.webapp.refresh_status",
            side_effect=_fake_refresh_status(status_map),
        ):
            dashboard = client.get("/")
            assert dashboard.status_code == 200
            body = dashboard.get_data(as_text=True)
            assert 'class="job-selector"' in body
            assert 'data-confirm="Kill selected jobs?"' in body
            assert 'data-confirm="Delete selected jobs?"' in body
            assert "Actions" not in body
            assert 'data-confirm="Delete this job?"' not in body

            with patch("smanager.webapp.cancel_job") as cancel_mock:
                response = client.post(
                    "/jobs/bulk",
                    data={"action": "kill", "job_uuid": ids["single_job_uuid"]},
                    follow_redirects=False,
                )
                assert response.status_code in {302, 303}
                cancel_mock.assert_called_once_with("12345")

            response = client.post(
                f"/jobs/{ids['dry_run_job_uuid']}/delete",
                follow_redirects=False,
            )
            assert response.status_code in {302, 303}


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


def test_serve_uses_waitress_by_default():
    """Normal web mode should use a WSGI server instead of Flask dev server."""
    fake_app = MagicMock()
    with patch("smanager.webapp.create_app", return_value=fake_app), patch(
        "smanager.webapp.waitress_serve"
    ) as waitress_mock, patch("smanager.webapp.console.print") as print_mock:
        serve(host="0.0.0.0", port=9000, debug=False)

    fake_app.run.assert_not_called()
    waitress_mock.assert_called_once_with(fake_app, host="0.0.0.0", port=9000)
    print_mock.assert_called_once()


def test_serve_uses_flask_debug_server_when_requested():
    """Debug mode should keep the Flask development server available."""
    fake_app = MagicMock()
    with patch("smanager.webapp.create_app", return_value=fake_app), patch(
        "smanager.webapp.waitress_serve"
    ) as waitress_mock, patch("smanager.webapp.console.print") as print_mock:
        serve(host="127.0.0.1", port=8000, debug=True)

    fake_app.run.assert_called_once_with(host="127.0.0.1", port=8000, debug=True)
    waitress_mock.assert_not_called()
    print_mock.assert_called_once()
