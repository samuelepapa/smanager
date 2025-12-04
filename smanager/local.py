"""Local sweep execution using tmux sessions."""

import json
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import SManagerConfig
from .sweep import load_sweep_function


class LocalSweep:  # pylint: disable=too-many-instance-attributes
    """Manages a parameter sweep running locally across multiple tmux sessions."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        script_path: str,
        sweep_file: str,
        sweep_function: str,
        base_args: Optional[List[str]] = None,
        experiment_name: Optional[str] = None,
        config: Optional[SManagerConfig] = None,
        arg_format: str = "--{key}={value}",
        workers: int = 1,
        gpus: Optional[str] = None,
        python_executable: str = "python",
        working_dir: Optional[str] = None,
        session_prefix: str = "sweep",
    ):
        """
        Initialize a local parameter sweep.

        Args:
            script_path: Path to the Python script to run.
            sweep_file: Path to Python file containing the sweep generator function.
            sweep_function: Name of the generator function in sweep_file.
            base_args: Base arguments always passed to the script.
            experiment_name: Name for the experiment.
            config: SManagerConfig for project settings.
            arg_format: Format string for arguments (default: "--{key}={value}").
            workers: Number of parallel tmux sessions/workers.
            gpus: Comma-separated GPU IDs to use (e.g., "0,1,2,3").
                  If not specified, GPUs are assigned round-robin style.
            python_executable: Python executable to use.
            working_dir: Working directory for the script.
            session_prefix: Prefix for tmux session names.
        """
        self.script_path = Path(script_path).resolve()
        self.sweep_file = Path(sweep_file).resolve()
        self.sweep_function_name = sweep_function
        self.base_args = base_args or []
        self.arg_format = arg_format
        self.workers = workers
        self.python_executable = python_executable
        self.working_dir = working_dir or str(Path.cwd())
        self.session_prefix = session_prefix

        # Parse GPU list
        if gpus:
            self.gpu_list = [g.strip() for g in gpus.split(",")]
        else:
            self.gpu_list = None

        # Load the sweep generator function
        self.sweep_function = load_sweep_function(
            str(self.sweep_file), self.sweep_function_name
        )

        # Generate experiment name
        if experiment_name:
            self.experiment_name = experiment_name
        else:
            parent_name = self.script_path.parent.name
            script_name = self.script_path.stem
            self.experiment_name = f"{parent_name}.{script_name}"

        self.config = config or SManagerConfig(self.script_path)

        # Generate UUID for this local sweep
        self.sweep_uuid = str(uuid.uuid4())

        # Store generated configurations
        self.param_sets: List[Dict[str, Any]] = []
        self.sweep_dir: Optional[Path] = None
        self.sessions: List[str] = []

    def _format_args(self, params: Dict[str, Any]) -> List[str]:
        """Format parameter dict into command line arguments."""
        args = []
        for key, value in params.items():
            arg = self.arg_format.format(key=key, value=value)
            args.append(arg)
        return args

    @staticmethod
    def _escape_for_bash_echo(text: str) -> str:
        """Escape a string for safe use inside bash double quotes."""
        # Escape backslashes first, then other special characters
        text = text.replace("\\", "\\\\")
        text = text.replace('"', '\\"')
        text = text.replace("$", "\\$")
        text = text.replace("`", "\\`")
        return text

    def generate_param_sets(self) -> List[Dict[str, Any]]:
        """
        Generate all parameter sets from the sweep generator.

        Returns:
            List of parameter dictionaries.
        """
        self.param_sets = []
        generator = self.sweep_function()

        for idx, params in enumerate(generator):
            if not isinstance(params, dict):
                type_name = type(params).__name__
                raise TypeError(
                    f"Sweep generator must yield dictionaries, "
                    f"got {type_name} at index {idx}"
                )
            self.param_sets.append(params)

        return self.param_sets

    def distribute_jobs(self) -> List[List[int]]:
        """
        Distribute job indices across workers.

        Returns:
            List of job index lists, one per worker.
        """
        if not self.param_sets:
            self.generate_param_sets()

        # Round-robin distribution
        worker_jobs: List[List[int]] = [[] for _ in range(self.workers)]
        for idx in range(len(self.param_sets)):
            worker_idx = idx % self.workers
            worker_jobs[worker_idx].append(idx)

        return worker_jobs

    def get_gpu_for_worker(self, worker_idx: int) -> Optional[str]:
        """Get the GPU ID(s) for a specific worker."""
        if self.gpu_list is None:
            return None

        if len(self.gpu_list) >= self.workers:
            # One or more GPUs per worker
            gpus_per_worker = len(self.gpu_list) // self.workers
            start = worker_idx * gpus_per_worker
            end = start + gpus_per_worker
            return ",".join(self.gpu_list[start:end])

        # Share GPUs across workers (round-robin)
        return self.gpu_list[worker_idx % len(self.gpu_list)]

    def _generate_worker_script(self, worker_idx: int, job_indices: List[int]) -> str:
        """
        Generate a bash script for a worker to run its assigned jobs.

        Args:
            worker_idx: Index of the worker.
            job_indices: List of job indices assigned to this worker.

        Returns:
            Bash script content.
        """
        lines = ["#!/bin/bash", ""]

        # Add preamble if available
        preamble = self.config.get_preamble()
        if preamble:
            lines.append("# === Preamble ===")
            lines.append(preamble.strip())
            lines.append("")

        # Set working directory
        lines.append(f"cd {self.working_dir}")
        lines.append("")

        # Set GPU environment variable
        gpu = self.get_gpu_for_worker(worker_idx)
        if gpu is not None:
            lines.append(f"export CUDA_VISIBLE_DEVICES={gpu}")
            lines.append(f'echo "Worker {worker_idx}: Using GPU(s) {gpu}"')
            lines.append("")

        # Add job execution
        lines.append(f"# === Running {len(job_indices)} jobs ===")
        lines.append("")

        for job_num, job_idx in enumerate(job_indices):
            params = self.param_sets[job_idx]
            sweep_args = self._format_args(params)
            all_args = self.base_args + sweep_args

            # Build the command
            cmd_parts = [self.python_executable, str(self.script_path)] + all_args
            cmd = " ".join(cmd_parts)

            lines.append(
                f"# Job {job_num + 1}/{len(job_indices)} (sweep index {job_idx})"
            )
            params_str = ", ".join(f"{k}={v}" for k, v in params.items())
            lines.append(f"# Params: {params_str}")
            escaped_params_str = self._escape_for_bash_echo(params_str)
            lines.append(
                f'echo ">>> [{job_num + 1}/{len(job_indices)}] '
                f'Running: {escaped_params_str}"'
            )
            lines.append(cmd)
            lines.append("EXIT_CODE=$?")
            lines.append("if [ $EXIT_CODE -ne 0 ]; then")
            lines.append(
                f'    echo ">>> Job {job_idx} failed with exit code $EXIT_CODE"'
            )
            lines.append("fi")
            lines.append("")

        lines.append(f'echo ">>> Worker {worker_idx} completed all jobs"')

        return "\n".join(lines)

    def save_scripts(self) -> List[Path]:
        """
        Save worker scripts to disk.

        Returns:
            List of paths to saved worker scripts.
        """
        if not self.param_sets:
            self.generate_param_sets()

        # Create sweep directory
        script_dir = self.config.get_script_dir()
        self.sweep_dir = script_dir / self.experiment_name / f"local_{self.sweep_uuid}"
        self.sweep_dir.mkdir(parents=True, exist_ok=True)

        # Create logs directory
        logs_dir = self.sweep_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Distribute jobs
        worker_jobs = self.distribute_jobs()

        # Generate and save worker scripts
        paths = []
        for worker_idx, job_indices in enumerate(worker_jobs):
            if not job_indices:
                continue

            script_content = self._generate_worker_script(worker_idx, job_indices)
            script_path = self.sweep_dir / f"worker_{worker_idx}.sh"
            script_path.write_text(script_content, encoding="utf-8")
            script_path.chmod(0o755)
            paths.append(script_path)

        # Save sweep mapping
        self._save_sweep_mapping(worker_jobs)

        return paths

    def _save_sweep_mapping(self, worker_jobs: List[List[int]]) -> None:
        """Save a JSON file with sweep configuration."""
        if not self.sweep_dir:
            return

        mapping_path = self.sweep_dir / "sweep.json"

        def make_serializable(obj: Any) -> Any:
            """Convert non-serializable objects to strings."""
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            if isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            if isinstance(obj, dict):
                return {str(k): make_serializable(v) for k, v in obj.items()}
            return str(obj)

        mapping = {
            "sweep_uuid": self.sweep_uuid,
            "type": "local",
            "script": str(self.script_path),
            "sweep_file": str(self.sweep_file),
            "sweep_function": self.sweep_function_name,
            "base_args": self.base_args,
            "arg_format": self.arg_format,
            "workers": self.workers,
            "gpus": ",".join(self.gpu_list) if self.gpu_list else None,
            "total_jobs": len(self.param_sets),
            "created_at": datetime.now().isoformat(),
            "jobs": {},
            "worker_assignments": {},
        }

        # Save job info
        for idx, params in enumerate(self.param_sets):
            mapping["jobs"][str(idx)] = {
                "index": idx,
                "params": make_serializable(params),
            }

        # Save worker assignments
        for worker_idx, job_indices in enumerate(worker_jobs):
            if job_indices:
                mapping["worker_assignments"][f"worker_{worker_idx}"] = {
                    "job_indices": job_indices,
                    "gpu": self.get_gpu_for_worker(worker_idx),
                    "session_name": f"{self.session_prefix}_{worker_idx}",
                }

        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)

    def _tmux_session_exists(self, session_name: str) -> bool:
        """Check if a tmux session already exists."""
        result = subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            capture_output=True,
            check=False,
        )
        return result.returncode == 0

    def _create_tmux_session(
        self, session_name: str, script_path: Path, log_path: Path
    ) -> bool:
        """
        Create a tmux session and run the worker script.

        Args:
            session_name: Name for the tmux session.
            script_path: Path to the worker script.
            log_path: Path for the output log.

        Returns:
            True if session was created successfully.
        """
        # Kill existing session if it exists
        if self._tmux_session_exists(session_name):
            subprocess.run(
                ["tmux", "kill-session", "-t", session_name],
                capture_output=True,
                check=False,
            )

        # Create new detached session running the script
        # Pipe output to a log file as well
        cmd = f"bash {script_path} 2>&1 | tee {log_path}"

        result = subprocess.run(
            [
                "tmux",
                "new-session",
                "-d",  # Detached
                "-s",
                session_name,  # Session name
                "bash",
                "-c",
                cmd,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        return result.returncode == 0

    def launch(self, dry_run: bool = False) -> List[str]:
        """
        Launch all worker sessions.

        Args:
            dry_run: If True, save scripts but don't launch tmux sessions.

        Returns:
            List of created tmux session names.
        """
        if not self.sweep_dir:
            self.save_scripts()

        self.sessions = []
        worker_jobs = self.distribute_jobs()
        logs_dir = self.sweep_dir / "logs"

        for worker_idx, job_indices in enumerate(worker_jobs):
            if not job_indices:
                continue

            session_name = f"{self.session_prefix}_{worker_idx}"
            script_path = self.sweep_dir / f"worker_{worker_idx}.sh"
            log_path = logs_dir / f"worker_{worker_idx}.log"

            if dry_run:
                self.sessions.append(session_name)
            else:
                if self._create_tmux_session(session_name, script_path, log_path):
                    self.sessions.append(session_name)

        return self.sessions

    def kill_sessions(self) -> int:
        """
        Kill all tmux sessions created by this sweep.

        Returns:
            Number of sessions killed.
        """
        killed = 0
        for worker_idx in range(self.workers):
            session_name = f"{self.session_prefix}_{worker_idx}"
            if self._tmux_session_exists(session_name):
                subprocess.run(
                    ["tmux", "kill-session", "-t", session_name],
                    capture_output=True,
                    check=False,
                )
                killed += 1
        return killed

    @staticmethod
    def list_sweep_sessions(prefix: str = "sweep") -> List[str]:
        """
        List all tmux sessions matching the sweep prefix.

        Args:
            prefix: Session name prefix to filter.

        Returns:
            List of matching session names.
        """
        result = subprocess.run(
            ["tmux", "list-sessions", "-F", "#{session_name}"],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            return []

        sessions = result.stdout.strip().split("\n")
        return [s for s in sessions if s.startswith(prefix)]

    def __len__(self) -> int:
        """Return the number of jobs in the sweep."""
        if self.param_sets:
            return len(self.param_sets)
        self.generate_param_sets()
        return len(self.param_sets)
