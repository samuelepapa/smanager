"""Flask web interface for browsing and managing SManager jobs."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from flask import Flask, abort, redirect, render_template, request, url_for

from .config import SManagerConfig, find_project_root
from .history import JobRecord, cancel_job, discover_jobs, find_job, refresh_status

PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"


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


def _make_app() -> Flask:
    return Flask(
        __name__,
        template_folder=str(TEMPLATE_DIR),
        static_folder=str(STATIC_DIR),
        static_url_path="/static",
    )


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

    def _load_records(show_dry_run: bool) -> List[JobRecord]:
        return discover_jobs(script_dir, include_dry_run=show_dry_run)

    @app.route("/")
    def dashboard():
        show_dry_run = request.args.get("show_dry_run", "0") in {"1", "true", "True"}
        records = _load_records(show_dry_run)
        records = refresh_status(records)
        records.sort(key=_sort_key, reverse=True)
        return render_template(
            "web/dashboard.html",
            records=records,
            show_dry_run=show_dry_run,
        )

    @app.route("/jobs/<job_uuid>")
    def job_detail(job_uuid: str):
        record = find_job(script_dir, job_uuid)
        if record is None:
            abort(404)
        refresh_status([record])
        active_tab = request.args.get("tab", "stdout")
        return render_template(
            "web/job_detail.html",
            record=record,
            active_tab=active_tab,
            job_error=None,
        )

    @app.route("/jobs/<job_uuid>/tab/<tab>")
    def job_tab(job_uuid: str, tab: str):
        record = find_job(script_dir, job_uuid)
        if record is None:
            abort(404)
        refresh_status([record])
        return render_template(
            "web/tab_content.html",
            record=record,
            active_tab=tab,
        )

    @app.route("/jobs/<job_uuid>/status")
    def job_status_fragment(job_uuid: str):
        record = find_job(script_dir, job_uuid)
        if record is None:
            abort(404)
        refresh_status([record])
        return render_template("web/status_fragment.html", record=record)

    @app.route("/jobs/<job_uuid>/kill", methods=["POST"])
    def job_kill(job_uuid: str):
        record = find_job(script_dir, job_uuid)
        if record is None:
            abort(404)
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

    return app


def serve(host: str = "127.0.0.1", port: int = 8000, debug: bool = False) -> None:
    """Run the dashboard server."""
    app = create_app()
    app.run(host=host, port=port, debug=debug)
