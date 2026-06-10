"""Flask web interface for browsing and managing SManager jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from flask import Flask, abort, redirect, render_template, request, url_for
from rich.console import Console

from .config import SManagerConfig, find_project_root
from .history import (
    JobRecord,
    cancel_job,
    delete_job,
    discover_jobs,
    find_job,
    refresh_status,
)

try:
    from waitress import serve as waitress_serve
except ImportError:  # pragma: no cover - dependency is installed in normal use
    waitress_serve = None

PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"
console = Console()


def _sort_key(record: JobRecord) -> str:
    return record.submitted_at or record.created_at or ""


def _status_text(record: JobRecord) -> str:
    if record.status_state:
        return f"{record.status_category} ({record.status_state})"
    if record.dry_run:
        return "dry-run"
    return record.status_category


def _status_class(record: JobRecord) -> str:
    if record.status_state:
        return record.status_category
    if record.dry_run:
        return "dry-run"
    return record.status_category


def _format_gpus(record: JobRecord) -> str:
    return "-" if record.gpus is None else str(record.gpus)


def _status_segments(records: List[JobRecord]) -> List[dict]:
    return [
        {
            "class": _status_class(record),
            "label": _status_text(record),
        }
        for record in records
    ]


def _make_dashboard_entries(records: List[JobRecord]) -> List[dict]:
    """Group sweep records into single dashboard entries."""
    entries: List[dict] = []
    sweep_entries: Dict[str, dict] = {}

    for record in records:
        if record.sweep_uuid:
            entry = sweep_entries.get(record.sweep_uuid)
            if entry is None:
                entry = {
                    "kind": "sweep",
                    "id": record.sweep_uuid,
                    "title": record.experiment_name,
                    "records": [],
                }
                sweep_entries[record.sweep_uuid] = entry
                entries.append(entry)
            entry["records"].append(record)
        else:
            entries.append(
                {
                    "kind": "job",
                    "id": record.job_uuid,
                    "title": record.experiment_name,
                    "records": [record],
                }
            )

    for entry in entries:
        entry["records"].sort(key=_sort_key, reverse=True)
        entry["primary"] = entry["records"][0]
        entry["segments"] = _status_segments(entry["records"])

    entries.sort(key=lambda entry: _sort_key(entry["primary"]), reverse=True)
    return entries


def _make_app() -> Flask:
    return Flask(
        __name__,
        template_folder=str(TEMPLATE_DIR),
        static_folder=str(STATIC_DIR),
        static_url_path="/static",
    )


def _show_dry_run_requested() -> bool:
    return request.args.get("show_dry_run", "0") in {"1", "true", "True"}


def _load_records(script_dir: Path, show_dry_run: bool) -> List[JobRecord]:
    return discover_jobs(script_dir, include_dry_run=show_dry_run)


def _find_record_or_404(script_dir: Path, job_uuid: str) -> JobRecord:
    record = find_job(script_dir, job_uuid)
    if record is None:
        abort(404)
    return record


def create_app(project_root: Optional[Path] = None) -> Flask:
    """Create the Flask app used by the web dashboard."""
    root = project_root or find_project_root(Path.cwd()) or Path.cwd()
    config = SManagerConfig(root)
    script_dir = config.get_script_dir()

    app = _make_app()
    app.config["SMANAGER_ROOT"] = str(root)
    app.config["SMANAGER_SCRIPT_DIR"] = str(script_dir)

    @app.context_processor
    def _inject_helpers() -> dict:
        return {
            "format_gpus": _format_gpus,
            "status_class": _status_class,
            "status_text": _status_text,
        }

    _register_routes(app, script_dir)
    return app


def _register_routes(app: Flask, script_dir: Path) -> None:
    """Register dashboard routes on an app instance."""

    @app.route("/")
    def dashboard():
        show_dry_run = _show_dry_run_requested()
        records = _load_records(script_dir, show_dry_run)
        records = refresh_status(records)
        entries = _make_dashboard_entries(records)
        return render_template(
            "web/dashboard.html",
            records=records,
            entries=entries,
            show_dry_run=show_dry_run,
        )

    @app.route("/jobs/<job_uuid>")
    def job_detail(job_uuid: str):
        record = _find_record_or_404(script_dir, job_uuid)
        refresh_status([record])
        active_tab = request.args.get("tab", "stdout")
        return render_template(
            "web/job_detail.html",
            record=record,
            active_tab=active_tab,
            job_error=None,
        )

    @app.route("/sweeps/<sweep_uuid>")
    def sweep_detail(sweep_uuid: str):
        show_dry_run = _show_dry_run_requested()
        records = [
            record
            for record in _load_records(script_dir, show_dry_run)
            if record.sweep_uuid == sweep_uuid
        ]
        if not records:
            abort(404)
        records = refresh_status(records)
        records.sort(key=lambda record: record.sweep_index or 0)
        param_keys = sorted(
            {key for record in records for key in record.hyperparameters.keys()}
        )
        return render_template(
            "web/sweep_detail.html",
            records=records,
            sweep_uuid=sweep_uuid,
            show_dry_run=show_dry_run,
            param_keys=param_keys,
        )

    @app.route("/jobs/<job_uuid>/tab/<tab>")
    def job_tab(job_uuid: str, tab: str):
        record = _find_record_or_404(script_dir, job_uuid)
        refresh_status([record])
        return render_template(
            "web/tab_content.html",
            record=record,
            active_tab=tab,
        )

    @app.route("/jobs/<job_uuid>/status")
    def job_status_fragment(job_uuid: str):
        record = _find_record_or_404(script_dir, job_uuid)
        refresh_status([record])
        return render_template("web/status_fragment.html", record=record)

    @app.route("/jobs/<job_uuid>/kill", methods=["POST"])
    def job_kill(job_uuid: str):
        record = _find_record_or_404(script_dir, job_uuid)
        refresh_status([record])
        active_tab = request.args.get("tab", "stdout")
        try:
            if record.can_kill and record.slurm_job_id:
                cancel_job(record.slurm_job_id)
        except OSError as exc:
            return render_template(
                "web/job_detail.html",
                record=record,
                active_tab=active_tab,
                job_error=f"Failed to kill job {record.slurm_job_id}: {exc}",
            )
        return redirect(url_for("job_detail", job_uuid=job_uuid, tab=active_tab))

    @app.route("/jobs/<job_uuid>/delete", methods=["POST"])
    def job_delete(job_uuid: str):
        record = _find_record_or_404(script_dir, job_uuid)
        delete_job(record, script_dir)
        return redirect(url_for("dashboard"))

    @app.route("/jobs/bulk", methods=["POST"])
    def jobs_bulk_action():
        action = request.form.get("action")
        job_uuids = request.form.getlist("job_uuid")
        records = [
            record
            for record in (find_job(script_dir, job_uuid) for job_uuid in job_uuids)
            if record is not None
        ]

        if action == "kill":
            for record in records:
                if record.can_kill and record.slurm_job_id:
                    cancel_job(record.slurm_job_id)
        elif action == "delete":
            for record in records:
                delete_job(record, script_dir)
        else:
            abort(400)

        show_dry_run = request.args.get("show_dry_run", "0")
        return redirect(url_for("dashboard", show_dry_run=show_dry_run))


def serve(host: str = "127.0.0.1", port: int = 8000, debug: bool = False) -> None:
    """Run the dashboard server."""
    app = create_app()
    if debug:
        console.print(
            f"[green]Starting SManager dashboard[/green] on "
            f"[cyan]http://{host}:{port}[/cyan] "
            "[dim](Flask development server)[/dim]"
        )
        app.run(host=host, port=port, debug=True)
        return

    if waitress_serve is None:
        raise RuntimeError(
            "waitress is required for the web dashboard. Install it with "
            "'pip install waitress' or 'pip install .'."
        )

    console.print(
        f"[green]Starting SManager dashboard[/green] on "
        f"[cyan]http://{host}:{port}[/cyan] [dim](Waitress)[/dim]"
    )
    waitress_serve(app, host=host, port=port)
