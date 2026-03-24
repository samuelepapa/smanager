"""Job history discovery and Slurm status helpers."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

JOB_STATUS_PENDING = {"PD", "CF"}
JOB_STATUS_RUNNING = {"R", "CG"}
JOB_STATUS_COMPLETED = {"CD"}
JOB_STATUS_FAILED = {"F", "CA", "TO", "NF", "PR", "BF", "OOM"}


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _format_datetime(value: Optional[str]) -> str:
    parsed = _parse_iso_datetime(value)
    if parsed is None:
        return "-"
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def _format_timedelta(delta: timedelta) -> str:
    total_seconds = int(delta.total_seconds())
    total_seconds = max(total_seconds, 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _current_runtime(submitted_at: Optional[str]) -> str:
    parsed = _parse_iso_datetime(submitted_at)
    if parsed is None:
        return "-"
    return _format_timedelta(datetime.now() - parsed)


def _normalize_path(
    value: Optional[str], base_dir: Optional[Path] = None
) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute() and base_dir is not None:
        path = (base_dir / path).resolve()
    return path


def _read_text(path: Optional[Path], default: str = "") -> str:
    if path is None or not path.exists():
        return default
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return default


def _job_status_category(state: Optional[str]) -> str:
    if not state:
        return "unknown"
    if state in JOB_STATUS_COMPLETED:
        return "done"
    if state in JOB_STATUS_FAILED:
        return "failed"
    if state in JOB_STATUS_PENDING or state in JOB_STATUS_RUNNING:
        return "running"
    return "running"


@dataclass
class JobRecord:  # pylint: disable=too-many-instance-attributes
    """Normalized record for a single job entry."""

    job_uuid: str
    kind: str
    experiment_name: str
    script_path: Path
    sbatch_path: Path
    output_path: Optional[Path]
    error_path: Optional[Path]
    script_args: List[str] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    slurm_options: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None
    submitted_at: Optional[str] = None
    slurm_job_id: Optional[str] = None
    dry_run: bool = False
    sweep_uuid: Optional[str] = None
    sweep_function: Optional[str] = None
    sweep_index: Optional[int] = None
    source_path: Optional[Path] = None
    status_state: Optional[str] = None
    status_time: Optional[str] = None
    status_node: Optional[str] = None
    status_exit_code: Optional[str] = None

    @property
    def partition(self) -> Optional[str]:
        return self.slurm_options.get("partition")

    @property
    def gpus(self) -> Optional[int]:
        value = self.slurm_options.get("gpus")
        return int(value) if value is not None else None

    @property
    def status_category(self) -> str:
        return _job_status_category(self.status_state)

    @property
    def run_time(self) -> str:
        if self.status_time and self.status_category in {"done", "failed"}:
            return self.status_time
        if self.status_time and self.status_category == "running":
            return self.status_time
        return _current_runtime(self.submitted_at)

    @property
    def run_started_at(self) -> str:
        return _format_datetime(self.submitted_at or self.created_at)

    @property
    def can_kill(self) -> bool:
        return bool(self.slurm_job_id) and not self.dry_run

    def load_stdout(self) -> str:
        return _read_text(self.output_path, default="[No stdout file found]")

    def load_stderr(self) -> str:
        return _read_text(self.error_path, default="[No stderr file found]")

    def load_sbatch(self) -> str:
        return _read_text(self.sbatch_path, default="[No sbatch file found]")

    def as_hyperparameter_items(self) -> List[tuple[str, str]]:
        items: List[tuple[str, str]] = [
            ("job_uuid", self.job_uuid),
            ("kind", self.kind),
            ("experiment_name", self.experiment_name),
            ("script_path", str(self.script_path)),
            ("sbatch_path", str(self.sbatch_path)),
            ("script_args", " ".join(self.script_args) if self.script_args else "-"),
            ("output_path", str(self.output_path) if self.output_path else "-"),
            ("error_path", str(self.error_path) if self.error_path else "-"),
            ("created_at", self.created_at or "-"),
            ("submitted_at", self.submitted_at or "-"),
            ("slurm_job_id", self.slurm_job_id or "-"),
            ("dry_run", str(self.dry_run)),
        ]
        if self.sweep_uuid:
            items.append(("sweep_uuid", self.sweep_uuid))
        if self.sweep_function:
            items.append(("sweep_function", self.sweep_function))
        if self.sweep_index is not None:
            items.append(("sweep_index", str(self.sweep_index)))
        for key, value in self.slurm_options.items():
            items.append((key, "-" if value is None else str(value)))
        for key, value in self.hyperparameters.items():
            items.append((key, "-" if value is None else str(value)))
        return items


def _load_squeue_status(job_ids: Sequence[str]) -> Dict[str, Dict[str, str]]:
    try:
        result = subprocess.run(
            ["squeue", "-j", ",".join(job_ids), "-h", "-o", "%A|%t|%M|%N|%j"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return {}

    if result.returncode != 0:
        return {}

    status_map: Dict[str, Dict[str, str]] = {}
    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 5:
            continue
        job_id, state, time_used, node, _name = parts[:5]
        status_map[job_id] = {
            "state": state,
            "time": time_used,
            "node": node or "-",
            "source": "squeue",
        }
    return status_map


def _load_sacct_status(job_ids: Sequence[str]) -> Dict[str, Dict[str, str]]:
    if not job_ids:
        return {}

    try:
        result = subprocess.run(
            [
                "sacct",
                "-j",
                ",".join(job_ids),
                "-n",
                "-P",
                "-o",
                "JobID,State,Elapsed,NodeList,ExitCode",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return {}

    if result.returncode != 0:
        return {}

    status_map: Dict[str, Dict[str, str]] = {}
    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 5:
            continue
        job_id, state, elapsed, node, exit_code = parts[:5]
        if "." in job_id:
            continue
        status_map[job_id] = {
            "state": state,
            "time": elapsed,
            "node": node or "-",
            "exit_code": exit_code,
            "source": "sacct",
        }

    return status_map


def _status_map_for_jobs(job_ids: Sequence[str]) -> Dict[str, Dict[str, str]]:
    if not job_ids:
        return {}

    status_map = _load_squeue_status(job_ids)
    missing_jobs = [job_id for job_id in job_ids if job_id not in status_map]
    if missing_jobs:
        status_map.update(_load_sacct_status(missing_jobs))
    return status_map


def refresh_status(records: List[JobRecord]) -> List[JobRecord]:
    """Populate job records with current Slurm status."""
    job_ids = [record.slurm_job_id for record in records if record.slurm_job_id]
    status_map = _status_map_for_jobs([job_id for job_id in job_ids if job_id])

    for record in records:
        if not record.slurm_job_id:
            continue
        status = status_map.get(record.slurm_job_id, {})
        record.status_state = status.get("state")
        record.status_time = status.get("time")
        record.status_node = status.get("node")
        record.status_exit_code = status.get("exit_code")

    return records


def _serialize_slurm_options(record: Any) -> Dict[str, Any]:
    return {
        "partition": getattr(record, "partition", None),
        "gpus": getattr(record, "gpus", None),
        "memory": getattr(record, "memory", None),
        "time": getattr(record, "time", None),
        "nodes": getattr(record, "nodes", None),
        "ntasks": getattr(record, "ntasks", None),
        "cpus_per_task": getattr(record, "cpus_per_task", None),
        "account": getattr(record, "account", None),
        "qos": getattr(record, "qos", None),
        "constraint": getattr(record, "constraint", None),
        "exclude": getattr(record, "exclude", None),
        "nodelist": getattr(record, "nodelist", None),
        "mail_type": getattr(record, "mail_type", None),
        "mail_user": getattr(record, "mail_user", None),
        "executable": getattr(record, "executable", None),
        "working_dir": str(getattr(record, "working_dir", "")) or None,
        "extra_sbatch_args": getattr(record, "extra_sbatch_args", None),
    }


def _make_job_record_from_manifest(manifest_path: Path) -> Optional[JobRecord]:
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if data.get("kind") != "job":
        return None

    script_path = _normalize_path(data.get("script_path") or data.get("script"))
    sbatch_path = _normalize_path(data.get("sbatch_path"))
    if script_path is None or sbatch_path is None:
        return None

    output_path = _normalize_path(data.get("output"))
    error_path = _normalize_path(data.get("error"))

    return JobRecord(
        job_uuid=data.get("job_uuid") or manifest_path.stem,
        kind="job",
        experiment_name=data.get("experiment_name", script_path.stem),
        script_path=script_path,
        sbatch_path=sbatch_path,
        output_path=output_path,
        error_path=error_path,
        script_args=list(data.get("script_args", [])),
        hyperparameters=dict(data.get("hyperparameters", {})),
        slurm_options=dict(data.get("slurm_options", {})),
        created_at=data.get("created_at"),
        submitted_at=data.get("submitted_at"),
        slurm_job_id=data.get("slurm_job_id"),
        dry_run=bool(data.get("dry_run", False)),
        source_path=manifest_path,
    )


def _make_job_records_from_sweep(manifest_path: Path) -> List[JobRecord]:
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    if data.get("kind") != "sweep":
        return []

    script_path = _normalize_path(data.get("script_path") or data.get("script"))
    if script_path is None:
        return []

    base_experiment = data.get("experiment_name") or script_path.stem
    sweep_uuid = data.get("sweep_uuid")
    sweep_function = data.get("sweep_function")
    root_created_at = data.get("created_at")
    root_dry_run = bool(data.get("dry_run", False))
    jobs_data = data.get("jobs", {})
    records: List[JobRecord] = []

    for job_uuid, job_info in jobs_data.items():
        sbatch_path = _normalize_path(job_info.get("sbatch_path"))
        if sbatch_path is None:
            sbatch_path = manifest_path.parent / f"{job_uuid}.sbatch"

        output_path = _normalize_path(job_info.get("output"))
        error_path = _normalize_path(job_info.get("error"))

        records.append(
            JobRecord(
                job_uuid=job_uuid,
                kind="sweep",
                experiment_name=base_experiment,
                script_path=script_path,
                sbatch_path=sbatch_path,
                output_path=output_path,
                error_path=error_path,
                script_args=list(job_info.get("script_args", [])),
                hyperparameters=dict(job_info.get("params", {})),
                slurm_options=dict(job_info.get("slurm_options", {})),
                created_at=job_info.get("created_at", root_created_at),
                submitted_at=job_info.get("submitted_at"),
                slurm_job_id=job_info.get("slurm_job_id"),
                dry_run=bool(job_info.get("dry_run", root_dry_run)),
                sweep_uuid=sweep_uuid,
                sweep_function=sweep_function,
                sweep_index=job_info.get("index"),
                source_path=manifest_path,
            )
        )

    return records


def discover_jobs(script_dir: Path, include_dry_run: bool = False) -> List[JobRecord]:
    """Discover all jobs stored under the scripts directory."""
    if not script_dir.exists():
        return []

    records: List[JobRecord] = []

    for manifest_path in script_dir.glob("**/*.json"):
        record = _make_job_record_from_manifest(manifest_path)
        if record is not None:
            records.append(record)

    for manifest_path in script_dir.glob("**/sweep.json"):
        records.extend(_make_job_records_from_sweep(manifest_path))

    if not include_dry_run:
        records = [record for record in records if not record.dry_run]

    return records


def find_job(script_dir: Path, job_uuid: str) -> Optional[JobRecord]:
    """Find a job record by UUID."""
    for record in discover_jobs(script_dir, include_dry_run=True):
        if record.job_uuid == job_uuid:
            return record
    return None


def cancel_job(job_id: str) -> subprocess.CompletedProcess[str]:
    """Cancel a Slurm job."""
    return subprocess.run(
        ["scancel", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
