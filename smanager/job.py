"""Job creation and submission handling."""

import subprocess
import uuid
from pathlib import Path
from typing import List, Optional

from .config import SManagerConfig
from .templates import SbatchTemplate


class SlurmJob:  # pylint: disable=too-many-instance-attributes
    """Represents a single Slurm job."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        script_path: str,
        script_args: Optional[List[str]] = None,
        experiment_name: Optional[str] = None,
        config: Optional[SManagerConfig] = None,
        *,
        partition: Optional[str] = None,
        gpus: Optional[int] = None,
        memory: Optional[str] = None,
        time: Optional[str] = None,
        nodes: Optional[int] = None,
        ntasks: Optional[int] = None,
        cpus_per_task: Optional[int] = None,
        output: Optional[str] = None,
        error: Optional[str] = None,
        mail_type: Optional[str] = None,
        mail_user: Optional[str] = None,
        account: Optional[str] = None,
        qos: Optional[str] = None,
        constraint: Optional[str] = None,
        exclude: Optional[str] = None,
        nodelist: Optional[str] = None,
        extra_sbatch_args: Optional[List[str]] = None,
        python_executable: str = "python",
        template: Optional[SbatchTemplate] = None,
        working_dir: Optional[str] = None,
    ):
        """
        Initialize a Slurm job.

        Args:
            script_path: Path to the Python script to run.
            script_args: Arguments to pass to the script.
            experiment_name: Name for the experiment (defaults to script name).
            config: SManagerConfig instance for project settings.
            partition: Slurm partition.
            gpus: Number of GPUs.
            memory: Memory allocation.
            time: Time limit.
            nodes: Number of nodes.
            ntasks: Number of tasks.
            cpus_per_task: CPUs per task.
            output: Output file path.
            error: Error file path.
            mail_type: Mail notification type.
            mail_user: Email address for notifications.
            account: Account for billing.
            qos: Quality of service.
            constraint: Node constraint.
            exclude: Nodes to exclude.
            nodelist: Specific nodes to use.
            extra_sbatch_args: Extra sbatch arguments.
            python_executable: Python executable to use.
            template: Custom sbatch template.
            working_dir: Working directory for the job.
        """
        self.script_path = Path(script_path).resolve()
        self.script_args = script_args or []

        # Use parent.script as experiment name if not provided
        # e.g., "fitting/train.py" -> "fitting.train"
        if experiment_name:
            self.experiment_name = experiment_name
        else:
            parent_name = self.script_path.parent.name
            script_name = self.script_path.stem
            self.experiment_name = f"{parent_name}.{script_name}"

        # Load or create config
        self.config = config or SManagerConfig(self.script_path)

        # Merge with config defaults (explicit args take precedence)
        defaults = self.config.defaults

        self.partition = partition or defaults.get("partition")
        self.gpus = gpus if gpus is not None else defaults.get("gpus")
        self.memory = memory or defaults.get("memory")
        self.time = time or defaults.get("time")
        self.nodes = nodes if nodes is not None else defaults.get("nodes")
        self.ntasks = ntasks if ntasks is not None else defaults.get("ntasks")
        self.cpus_per_task = (
            cpus_per_task
            if cpus_per_task is not None
            else defaults.get("cpus_per_task")
        )
        self.output = output or defaults.get("output")
        self.error = error or defaults.get("error")
        self.mail_type = mail_type or defaults.get("mail_type")
        self.mail_user = mail_user or defaults.get("mail_user")
        self.account = account or defaults.get("account")
        self.qos = qos or defaults.get("qos")
        self.constraint = constraint or defaults.get("constraint")
        self.exclude = exclude or defaults.get("exclude")
        self.nodelist = nodelist or defaults.get("nodelist")
        self.extra_sbatch_args = extra_sbatch_args or defaults.get(
            "extra_sbatch_args", []
        )
        self.python_executable = python_executable or defaults.get(
            "python_executable", "python"
        )

        self.working_dir = Path(working_dir).resolve() if working_dir else Path.cwd()
        self.template = template or SbatchTemplate()

        # Generate UUID for this job
        self.job_uuid = str(uuid.uuid4())

        # Generated script path (set after generate())
        self.sbatch_script_path: Optional[Path] = None
        self.job_id: Optional[str] = None

    def generate_script(self, job_name: Optional[str] = None) -> str:
        """
        Generate the sbatch script content.

        Args:
            job_name: Custom job name (uses UUID if not provided).

        Returns:
            The sbatch script content as a string.
        """
        if job_name is None:
            job_name = self.job_uuid

        script_args_str = " ".join(self.script_args) if self.script_args else ""

        return self.template.render(
            script_path=str(self.script_path),
            job_name=job_name,
            working_dir=str(self.working_dir),
            script_args=script_args_str,
            preamble=self.config.get_preamble(),
            python_executable=self.python_executable,
            partition=self.partition,
            gpus=self.gpus,
            memory=self.memory,
            time=self.time,
            nodes=self.nodes,
            ntasks=self.ntasks,
            cpus_per_task=self.cpus_per_task,
            output=self.output,
            error=self.error,
            mail_type=self.mail_type,
            mail_user=self.mail_user,
            account=self.account,
            qos=self.qos,
            constraint=self.constraint,
            exclude=self.exclude,
            nodelist=self.nodelist,
            extra_sbatch_args=self.extra_sbatch_args,
        )

    def save_script(self, script_dir: Optional[Path] = None) -> Path:
        """
        Save the sbatch script to disk.

        Args:
            script_dir: Directory to save the script.

        Returns:
            Path to the saved script file.
        """
        if script_dir is None:
            script_dir = self.config.get_script_dir()

        # Create experiment subdirectory
        experiment_dir = script_dir / self.experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create logs subdirectory
        logs_dir = experiment_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Set default log paths if not specified
        if self.output is None:
            self.output = str(logs_dir / f"{self.job_uuid}.out")
        if self.error is None:
            self.error = str(logs_dir / f"{self.job_uuid}.err")

        script_content = self.generate_script()
        script_path = experiment_dir / f"{self.job_uuid}.sbatch"

        script_path.write_text(script_content, encoding="utf-8")
        self.sbatch_script_path = script_path

        return script_path

    def submit(self, dry_run: bool = False) -> Optional[str]:
        """
        Submit the job to Slurm.

        Args:
            dry_run: If True, just print what would be done without submitting.

        Returns:
            Job ID if submitted successfully, None otherwise.
        """
        if self.sbatch_script_path is None:
            self.save_script()

        cmd = ["sbatch", str(self.sbatch_script_path)]

        if dry_run:
            return None

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # Parse job ID from output like "Submitted batch job 12345"
            output = result.stdout.strip()
            if "Submitted batch job" in output:
                self.job_id = output.split()[-1]
                return self.job_id
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Failed to submit job: {exc.stderr}") from exc
        except FileNotFoundError as exc:
            raise RuntimeError("sbatch command not found. Is Slurm installed?") from exc

        return None


def create_job_from_args(
    script_path: str, script_args: List[str], **slurm_options
) -> SlurmJob:
    """
    Factory function to create a SlurmJob from CLI arguments.

    Args:
        script_path: Path to Python script.
        script_args: Arguments for the script.
        **slurm_options: Slurm configuration options.

    Returns:
        Configured SlurmJob instance.
    """
    return SlurmJob(script_path=script_path, script_args=script_args, **slurm_options)
