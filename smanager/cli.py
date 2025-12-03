"""Command-line interface for Slurm Manager."""

# pylint: disable=too-many-lines

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import click
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .config import SManagerConfig
from .job import SlurmJob
from .local import LocalSweep
from .sweep import Sweep

console = Console()


def parse_extra_args(args: Tuple[str, ...]) -> List[str]:
    """Parse extra arguments passed to the script."""
    return list(args)


@click.group()
@click.version_option(version="0.2.0", prog_name="smanager")
def cli():
    """
    Slurm Manager - A CLI tool for managing Slurm jobs and parameter sweeps.

    \b
    Examples:
        smanager run train.py --gpus 4 --memory 32G
        smanager run train.py --gpus 2 --dry-run --show
        smanager sweep train.py sweeps.py my_sweep --gpus 4
        smanager sweep train.py sweeps.py grid --dry-run
        smanager kill                  # cancel last sweep
        smanager kill --dry-run        # show what would be cancelled
        smanager history               # list recent sweeps
    """


@cli.command()
@click.argument("script", type=click.Path(exists=True))
@click.option("--name", "-n", help="Experiment name (defaults to script name)")
@click.option("--gpus", "-g", type=int, help="Number of GPUs")
@click.option("--memory", "-m", help="Memory (e.g., 32G, 64000M)")
@click.option("--time", "-t", "time_limit", help="Time limit (e.g., 24:00:00)")
@click.option("--partition", "-p", help="Slurm partition")
@click.option("--nodes", type=int, help="Number of nodes")
@click.option("--ntasks", type=int, help="Number of tasks")
@click.option("--cpus-per-task", "-c", type=int, help="CPUs per task")
@click.option("--account", "-A", help="Account/project for billing")
@click.option("--qos", help="Quality of Service")
@click.option("--constraint", help="Node feature constraint")
@click.option("--exclude", help="Nodes to exclude")
@click.option("--nodelist", help="Specific nodes to use")
@click.option("--output", "-o", help="Output file pattern")
@click.option("--error", "-e", help="Error file pattern")
@click.option("--mail-type", help="Mail notification type (BEGIN,END,FAIL,ALL)")
@click.option("--mail-user", help="Email address for notifications")
@click.option(
    "--python", "python_executable", default="python", help="Python executable"
)
@click.option("--working-dir", "-w", type=click.Path(), help="Working directory")
@click.option("--dry-run", "-d", is_flag=True, help="Generate script but do not submit")
@click.option("--show", "-s", is_flag=True, help="Show generated script")
@click.option(
    "--sbatch-arg", multiple=True, help="Extra sbatch arguments (can be repeated)"
)
@click.argument("script_args", nargs=-1, type=click.UNPROCESSED)
def run(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    script: str,
    name: Optional[str],
    gpus: Optional[int],
    memory: Optional[str],
    time_limit: Optional[str],
    partition: Optional[str],
    nodes: Optional[int],
    ntasks: Optional[int],
    cpus_per_task: Optional[int],
    account: Optional[str],
    qos: Optional[str],
    constraint: Optional[str],
    exclude: Optional[str],
    nodelist: Optional[str],
    output: Optional[str],
    error: Optional[str],
    mail_type: Optional[str],
    mail_user: Optional[str],
    python_executable: str,
    working_dir: Optional[str],
    dry_run: bool,
    show: bool,
    sbatch_arg: Tuple[str, ...],
    script_args: Tuple[str, ...],
):
    """
    Run a Python script as a Slurm job.

    \b
    SCRIPT: Path to the Python script to run.
    SCRIPT_ARGS: Additional arguments to pass to the script (after --).

    \b
    Examples:
        smanager run train.py --gpus 4 --memory 32G
        smanager run train.py -- --lr 0.01 --epochs 100
        smanager run train.py -g 2 -t 12:00:00 --dry-run --show
    """
    try:
        job = SlurmJob(
            script_path=script,
            script_args=parse_extra_args(script_args),
            experiment_name=name,
            partition=partition,
            gpus=gpus,
            memory=memory,
            time=time_limit,
            nodes=nodes,
            ntasks=ntasks,
            cpus_per_task=cpus_per_task,
            output=output,
            error=error,
            mail_type=mail_type,
            mail_user=mail_user,
            account=account,
            qos=qos,
            constraint=constraint,
            exclude=exclude,
            nodelist=nodelist,
            extra_sbatch_args=list(sbatch_arg) if sbatch_arg else None,
            python_executable=python_executable,
            working_dir=working_dir,
        )

        script_path = job.save_script()

        if show:
            script_content = script_path.read_text(encoding="utf-8")
            console.print(
                Panel(
                    Syntax(script_content, "bash", theme="monokai"),
                    title=f"[bold blue]{script_path.name}[/bold blue]",
                    border_style="blue",
                )
            )

        console.print(f"[green]✓[/green] Script saved to: [cyan]{script_path}[/cyan]")
        console.print(f"[dim]  Job UUID: {job.job_uuid}[/dim]")

        if job.config.config_dir:
            console.print(f"[dim]  Using config from: {job.config.config_dir}[/dim]")

        if dry_run:
            console.print("\n[yellow]⚠[/yellow] Dry run mode - job not submitted")
            console.print(f"[dim]  Would run: sbatch {script_path}[/dim]")
            if not show:
                console.print("[dim]  Use --show to see the generated script[/dim]")
        else:
            job_id = job.submit()
            if job_id:
                console.print(
                    f"[green]✓[/green] Job submitted with ID: "
                    f"[bold green]{job_id}[/bold green]"
                )
            else:
                console.print("[red]✗[/red] Failed to submit job")
                sys.exit(1)

    except RuntimeError as exc:
        console.print(f"[red]✗ Error:[/red] {exc}")
        sys.exit(1)


@cli.command()
@click.argument("script", type=click.Path(exists=True))
@click.argument("sweep_file", type=click.Path(exists=True))
@click.argument("sweep_function")
@click.option("--name", "-n", help="Experiment name (defaults to script_sweep)")
@click.option("--gpus", "-g", type=int, help="Number of GPUs")
@click.option("--memory", "-m", help="Memory (e.g., 32G)")
@click.option("--time", "-t", "time_limit", help="Time limit (e.g., 24:00:00)")
@click.option("--partition", "-p", help="Slurm partition")
@click.option("--nodes", type=int, help="Number of nodes")
@click.option("--ntasks", type=int, help="Number of tasks")
@click.option("--cpus-per-task", "-c", type=int, help="CPUs per task")
@click.option("--account", "-A", help="Account/project for billing")
@click.option("--qos", help="Quality of Service")
@click.option("--constraint", help="Node feature constraint")
@click.option("--exclude", help="Nodes to exclude")
@click.option("--nodelist", help="Specific nodes to use")
@click.option("--output", "-o", help="Output file pattern")
@click.option("--error", "-e", help="Error file pattern")
@click.option("--mail-type", help="Mail notification type")
@click.option("--mail-user", help="Email address for notifications")
@click.option(
    "--python", "python_executable", default="python", help="Python executable"
)
@click.option("--working-dir", "-w", type=click.Path(), help="Working directory")
@click.option(
    "--dry-run", "-d", is_flag=True, help="Generate scripts but do not submit"
)
@click.option(
    "--delay", type=float, default=0.0, help="Delay between job submissions (seconds)"
)
@click.option(
    "--arg-format",
    default="--{key}={value}",
    help="Format for sweep args (default: --{key}={value})",
)
@click.option("--sbatch-arg", multiple=True, help="Extra sbatch arguments")
@click.argument("base_args", nargs=-1, type=click.UNPROCESSED)
def sweep(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    script: str,
    sweep_file: str,
    sweep_function: str,
    name: Optional[str],
    gpus: Optional[int],
    memory: Optional[str],
    time_limit: Optional[str],
    partition: Optional[str],
    nodes: Optional[int],
    ntasks: Optional[int],
    cpus_per_task: Optional[int],
    account: Optional[str],
    qos: Optional[str],
    constraint: Optional[str],
    exclude: Optional[str],
    nodelist: Optional[str],
    output: Optional[str],
    error: Optional[str],
    mail_type: Optional[str],
    mail_user: Optional[str],
    python_executable: str,
    working_dir: Optional[str],
    dry_run: bool,
    delay: float,
    arg_format: str,
    sbatch_arg: Tuple[str, ...],
    base_args: Tuple[str, ...],
):
    """
    Run a parameter sweep with multiple Slurm jobs.

    \b
    SCRIPT: Path to the Python script to run.
    SWEEP_FILE: Python file containing the sweep generator function.
    SWEEP_FUNCTION: Name of the generator function that yields parameter dicts.
    BASE_ARGS: Base arguments always passed to the script (after --).

    \b
    Examples:
        smanager sweep train.py sweeps.py learning_rate_sweep --gpus 4
        smanager sweep train.py sweeps.py grid_search -g 2 -t 12:00:00
        smanager sweep train.py sweeps.py my_sweep -- --epochs 100
    """
    try:
        sweep_obj = Sweep(
            script_path=script,
            sweep_file=sweep_file,
            sweep_function=sweep_function,
            base_args=parse_extra_args(base_args),
            experiment_name=name,
            arg_format=arg_format,
            partition=partition,
            gpus=gpus,
            memory=memory,
            time_limit=time_limit,
            nodes=nodes,
            ntasks=ntasks,
            cpus_per_task=cpus_per_task,
            output=output,
            error=error,
            mail_type=mail_type,
            mail_user=mail_user,
            account=account,
            qos=qos,
            constraint=constraint,
            exclude=exclude,
            nodelist=nodelist,
            extra_sbatch_args=list(sbatch_arg) if sbatch_arg else None,
            python_executable=python_executable,
            working_dir=working_dir,
        )

        sweep_obj.generate_jobs()
        _display_sweep_info(sweep_obj, script, sweep_file, sweep_function)
        _save_and_submit_sweep(sweep_obj, dry_run, delay)

    except FileNotFoundError as exc:
        console.print(f"[red]✗ File not found:[/red] {exc}")
        sys.exit(1)
    except AttributeError as exc:
        console.print(f"[red]✗ Function not found:[/red] {exc}")
        sys.exit(1)
    except TypeError as exc:
        console.print(f"[red]✗ Invalid sweep function:[/red] {exc}")
        sys.exit(1)


def _display_sweep_info(
    sweep_obj: Sweep, script: str, sweep_file: str, sweep_function: str
) -> None:
    """Display sweep information."""
    console.print(
        Panel(
            f"[bold]Script:[/bold] {script}\n"
            f"[bold]Sweep File:[/bold] {sweep_file}\n"
            f"[bold]Sweep Function:[/bold] {sweep_function}\n"
            f"[bold]Sweep UUID:[/bold] {sweep_obj.sweep_uuid}\n"
            f"[bold]Total Jobs:[/bold] {len(sweep_obj.jobs)}",
            title="[bold blue]Parameter Sweep[/bold blue]",
            border_style="blue",
        )
    )

    if sweep_obj.jobs:
        table = Table(title="Sweep Parameters (preview)")
        first_params = sweep_obj.jobs[0].sweep_params
        for key in first_params.keys():
            table.add_column(key, style="cyan")

        preview_count = min(10, len(sweep_obj.jobs))
        for job in sweep_obj.jobs[:preview_count]:
            row = [str(v) for v in job.sweep_params.values()]
            table.add_row(*row)

        if len(sweep_obj.jobs) > preview_count:
            table.add_row(*["..." for _ in first_params])

        console.print(table)


def _save_and_submit_sweep(sweep_obj: Sweep, dry_run: bool, delay: float) -> None:
    """Save sweep scripts and optionally submit them."""
    scripts = sweep_obj.save_scripts()
    console.print(f"\n[green]✓[/green] Generated {len(scripts)} job scripts")
    console.print(f"[dim]  Scripts saved in: {sweep_obj.sweep_dir}[/dim]")
    console.print(
        f"[dim]  Parameter mapping: {sweep_obj.sweep_dir / 'sweep.json'}[/dim]"
    )

    if sweep_obj.config.config_dir:
        console.print(f"[dim]  Using config from: {sweep_obj.config.config_dir}[/dim]")

    if dry_run:
        console.print("\n[yellow]⚠[/yellow] Dry run mode - jobs not submitted")
        return

    console.print("\n[bold]Submitting jobs...[/bold]")
    job_ids = sweep_obj.submit_all(delay=delay)

    submitted = sum(1 for jid in job_ids if jid is not None)
    console.print(f"[green]✓[/green] Submitted {submitted}/{len(job_ids)} jobs")

    if submitted > 0:
        console.print("\n[bold]Job IDs:[/bold]")
        for job, job_id in zip(sweep_obj.jobs, job_ids):
            if job_id:
                params_str = ", ".join(f"{k}={v}" for k, v in job.sweep_params.items())
                console.print(f"  [green]{job_id}[/green]: {params_str}")


@cli.command("local")
@click.argument("script", type=click.Path(exists=True))
@click.argument("sweep_file", type=click.Path(exists=True))
@click.argument("sweep_function")
@click.option("--name", "-n", help="Experiment name (defaults to script_sweep)")
@click.option(
    "--workers",
    "-w",
    type=int,
    default=1,
    help="Number of parallel tmux sessions/workers",
)
@click.option(
    "--gpus",
    "-g",
    help="Comma-separated GPU IDs to use (e.g., '0,1,2,3')",
)
@click.option(
    "--python", "python_executable", default="python", help="Python executable"
)
@click.option("--working-dir", type=click.Path(), help="Working directory")
@click.option(
    "--session-prefix",
    "-s",
    default="sweep",
    help="Prefix for tmux session names (default: sweep)",
)
@click.option(
    "--dry-run", "-d", is_flag=True, help="Generate scripts but do not launch tmux"
)
@click.option("--show", is_flag=True, help="Show generated worker scripts")
@click.option(
    "--arg-format",
    default="--{key}={value}",
    help="Format for sweep args (default: --{key}={value})",
)
@click.argument("base_args", nargs=-1, type=click.UNPROCESSED)
def local_sweep(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    script: str,
    sweep_file: str,
    sweep_function: str,
    name: Optional[str],
    workers: int,
    gpus: Optional[str],
    python_executable: str,
    working_dir: Optional[str],
    session_prefix: str,
    dry_run: bool,
    show: bool,
    arg_format: str,
    base_args: Tuple[str, ...],
):
    """
    Run a parameter sweep locally using tmux sessions.

    This command is useful when you have an allocated node with multiple GPUs
    and want to run a sweep without submitting multiple Slurm jobs. Each worker
    runs in its own tmux session with assigned GPUs.

    \b
    SCRIPT: Path to the Python script to run.
    SWEEP_FILE: Python file containing the sweep generator function.
    SWEEP_FUNCTION: Name of the generator function that yields parameter dicts.
    BASE_ARGS: Base arguments always passed to the script (after --).

    \b
    Examples:
        # Run sweep with 4 workers, auto-assign GPUs
        smanager local train.py sweeps.py my_sweep --workers 4 --gpus 0,1,2,3

        # Run with specific GPU assignment
        smanager local train.py sweeps.py grid --workers 2 --gpus 0,1

        # Dry run to see generated scripts
        smanager local train.py sweeps.py my_sweep -w 4 -g 0,1,2,3 --dry-run --show

        # With base arguments
        smanager local train.py sweeps.py my_sweep -w 4 -- --epochs 100
    """
    try:
        local_obj = LocalSweep(
            script_path=script,
            sweep_file=sweep_file,
            sweep_function=sweep_function,
            base_args=parse_extra_args(base_args),
            experiment_name=name,
            arg_format=arg_format,
            workers=workers,
            gpus=gpus,
            python_executable=python_executable,
            working_dir=working_dir,
            session_prefix=session_prefix,
        )

        local_obj.generate_param_sets()
        _display_local_sweep_info(local_obj, script, sweep_file, sweep_function)

        scripts = local_obj.save_scripts()

        if show:
            for script_path in scripts:
                script_content = script_path.read_text(encoding="utf-8")
                console.print(
                    Panel(
                        Syntax(script_content, "bash", theme="monokai"),
                        title=f"[bold blue]{script_path.name}[/bold blue]",
                        border_style="blue",
                    )
                )

        console.print(f"\n[green]✓[/green] Generated {len(scripts)} worker scripts")
        console.print(f"[dim]  Scripts saved in: {local_obj.sweep_dir}[/dim]")

        if local_obj.config.config_dir:
            console.print(
                f"[dim]  Using config from: {local_obj.config.config_dir}[/dim]"
            )

        if dry_run:
            console.print(
                "\n[yellow]⚠[/yellow] Dry run mode - tmux sessions not launched"
            )
            console.print("[dim]  Use --show to see the generated worker scripts[/dim]")
            return

        console.print("\n[bold]Launching tmux sessions...[/bold]")
        sessions = local_obj.launch()

        console.print(f"[green]✓[/green] Launched {len(sessions)} tmux sessions")
        console.print("\n[bold]Sessions:[/bold]")
        for idx, session in enumerate(sessions):
            gpu = local_obj.get_gpu_for_worker(idx)
            gpu_info = f" (GPU {gpu})" if gpu else ""
            console.print(f"  [cyan]{session}[/cyan]{gpu_info}")

        console.print(
            "\n[dim]Attach to a session with: tmux attach -t <session_name>[/dim]"
        )
        console.print(
            f"[dim]Kill all sessions with: smanager local-kill {session_prefix}[/dim]"
        )

    except FileNotFoundError as exc:
        console.print(f"[red]✗ File not found:[/red] {exc}")
        sys.exit(1)
    except AttributeError as exc:
        console.print(f"[red]✗ Function not found:[/red] {exc}")
        sys.exit(1)
    except TypeError as exc:
        console.print(f"[red]✗ Invalid sweep function:[/red] {exc}")
        sys.exit(1)


def _display_local_sweep_info(
    local_obj: LocalSweep, script: str, sweep_file: str, sweep_function: str
) -> None:
    """Display local sweep information."""
    gpu_info = local_obj.gpu_list
    gpu_str = ", ".join(gpu_info) if gpu_info else "auto"

    console.print(
        Panel(
            f"[bold]Script:[/bold] {script}\n"
            f"[bold]Sweep File:[/bold] {sweep_file}\n"
            f"[bold]Sweep Function:[/bold] {sweep_function}\n"
            f"[bold]Sweep UUID:[/bold] {local_obj.sweep_uuid}\n"
            f"[bold]Total Jobs:[/bold] {len(local_obj.param_sets)}\n"
            f"[bold]Workers:[/bold] {local_obj.workers}\n"
            f"[bold]GPUs:[/bold] {gpu_str}",
            title="[bold blue]Local Parameter Sweep[/bold blue]",
            border_style="blue",
        )
    )

    if local_obj.param_sets:
        table = Table(title="Sweep Parameters (preview)")
        first_params = local_obj.param_sets[0]
        for key in first_params.keys():
            table.add_column(key, style="cyan")

        preview_count = min(10, len(local_obj.param_sets))
        for params in local_obj.param_sets[:preview_count]:
            row = [str(v) for v in params.values()]
            table.add_row(*row)

        if len(local_obj.param_sets) > preview_count:
            table.add_row(*["..." for _ in first_params])

        console.print(table)

    # Show job distribution
    jobs_per_worker = len(local_obj.param_sets) // local_obj.workers
    remainder = len(local_obj.param_sets) % local_obj.workers

    dist_table = Table(title="Job Distribution")
    dist_table.add_column("Worker", style="cyan")
    dist_table.add_column("Jobs", style="green")
    dist_table.add_column("GPU(s)", style="yellow")

    for w in range(local_obj.workers):
        job_count = jobs_per_worker + (1 if w < remainder else 0)
        gpu = local_obj.get_gpu_for_worker(w) or "-"
        dist_table.add_row(f"worker_{w}", str(job_count), gpu)

    console.print(dist_table)


@cli.command("local-kill")
@click.argument("prefix", default="sweep")
@click.option(
    "--dry-run",
    "-d",
    is_flag=True,
    help="Show what would be killed without actually killing",
)
def local_kill(prefix: str, dry_run: bool):
    """
    Kill tmux sessions created by local sweep.

    \b
    PREFIX: Session name prefix to kill (default: sweep).

    \b
    Examples:
        smanager local-kill              # Kill all 'sweep_*' sessions
        smanager local-kill mysweep      # Kill all 'mysweep_*' sessions
        smanager local-kill --dry-run    # Show what would be killed
    """
    sessions = LocalSweep.list_sweep_sessions(prefix)

    if not sessions:
        console.print(
            f"[yellow]⚠[/yellow] No tmux sessions found with prefix '{prefix}'"
        )
        return

    table = Table(title="Sessions to Kill" if not dry_run else "Sessions (Dry Run)")
    table.add_column("Session", style="cyan")
    table.add_column("Status", style="yellow")

    for session in sessions:
        status = "Will kill" if not dry_run else "Would kill"
        table.add_row(session, status)

    console.print(table)

    if dry_run:
        console.print(
            f"\n[yellow]⚠[/yellow] Dry run - would kill {len(sessions)} sessions"
        )
        return

    killed = 0
    for session in sessions:
        result = subprocess.run(
            ["tmux", "kill-session", "-t", session],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            killed += 1

    console.print(f"\n[green]✓[/green] Killed {killed}/{len(sessions)} sessions")


@cli.command("local-list")
@click.argument("prefix", default="sweep")
def local_list(prefix: str):
    """
    List active tmux sessions from local sweeps.

    \b
    PREFIX: Session name prefix to filter (default: sweep).
    """
    sessions = LocalSweep.list_sweep_sessions(prefix)

    if not sessions:
        console.print(f"[dim]No tmux sessions found with prefix '{prefix}'[/dim]")
        return

    table = Table(title=f"Active Sessions (prefix: {prefix})")
    table.add_column("Session", style="cyan")

    for session in sessions:
        table.add_row(session)

    console.print(table)
    console.print("\n[dim]Attach with: tmux attach -t <session_name>[/dim]")


@cli.command()
@click.argument("path", type=click.Path(), default=".")
def init(path: str):
    """
    Initialize a .smanager directory with default configuration.

    \b
    PATH: Directory to initialize (defaults to current directory).
    """
    try:
        base_path = Path(path).resolve()
        smanager_dir = base_path / ".smanager"

        if smanager_dir.exists():
            console.print(
                f"[yellow]⚠[/yellow] .smanager already exists at {smanager_dir}"
            )
            if not click.confirm("Overwrite existing configuration?"):
                return

        smanager_dir.mkdir(parents=True, exist_ok=True)
        (smanager_dir / "scripts").mkdir(exist_ok=True)

        _write_default_config(smanager_dir)
        _write_default_preamble(smanager_dir)
        _write_example_sweeps(smanager_dir)

        console.print(
            f"[green]✓[/green] Initialized .smanager at [cyan]{smanager_dir}[/cyan]"
        )
        console.print("\nCreated files:")
        console.print("  [cyan]config.yaml[/cyan]  - Edit to set default Slurm options")
        console.print(
            "  [cyan]preamble.sh[/cyan]  - Edit to add environment setup commands"
        )
        console.print("  [cyan]sweeps.py[/cyan]    - Example sweep generator functions")
        console.print(
            "  [cyan]scripts/[/cyan]     - Generated sbatch scripts will be saved here"
        )

    except OSError as exc:
        console.print(f"[red]✗ Error:[/red] {exc}")
        sys.exit(1)


def _write_default_config(smanager_dir: Path) -> None:
    """Write default config.yaml."""
    config_content = """\
# Slurm Manager Default Configuration
# These values are used when not specified on the command line

# partition: gpu
# gpus: 1
# memory: 32G
# time: "24:00:00"
# cpus_per_task: 4
# account: your_account
# python_executable: python

# Extra sbatch arguments (list)
# extra_sbatch_args:
#   - "--exclusive"
#   - "--requeue"
"""
    (smanager_dir / "config.yaml").write_text(config_content, encoding="utf-8")


def _write_default_preamble(smanager_dir: Path) -> None:
    """Write default preamble.sh."""
    preamble_content = """\
# Preamble - Add environment setup commands here
# These commands run before your Python script

# Example: Activate conda environment
# source ~/miniconda3/bin/activate
# conda activate myenv

# Example: Load modules
# module load cuda/11.8
# module load python/3.10

# Example: Set environment variables
# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# export DATA_PATH=/path/to/data
# export WANDB_PROJECT=my_project

# Example: Navigate to project
# cd $SLURM_SUBMIT_DIR
"""
    (smanager_dir / "preamble.sh").write_text(preamble_content, encoding="utf-8")


def _write_example_sweeps(smanager_dir: Path) -> None:
    """Write example sweeps.py."""
    sweeps_content = '''\
"""
Example sweep definitions for smanager.

Each sweep function should be a generator that yields dictionaries.
Each dictionary contains parameter names and values for one job.
"""


def learning_rate_sweep():
    """Simple learning rate sweep."""
    for lr in [0.1, 0.01, 0.001, 0.0001]:
        yield {"lr": lr}


def grid_search():
    """Grid search over learning rate and batch size."""
    learning_rates = [0.1, 0.01, 0.001]
    batch_sizes = [32, 64, 128]

    for lr in learning_rates:
        for batch_size in batch_sizes:
            yield {
                "lr": lr,
                "batch_size": batch_size,
            }


def model_comparison():
    """Compare different model architectures."""
    models = ["resnet18", "resnet50", "vgg16", "efficientnet"]

    for model in models:
        yield {
            "model": model,
            "pretrained": True,
        }


def custom_sweep():
    """
    Example of a more complex sweep with conditional parameters.

    You can use any Python logic to generate parameter combinations.
    """
    configs = [
        {"model": "small", "hidden_dim": 256, "num_layers": 2},
        {"model": "medium", "hidden_dim": 512, "num_layers": 4},
        {"model": "large", "hidden_dim": 1024, "num_layers": 6},
    ]

    for config in configs:
        for dropout in [0.1, 0.3, 0.5]:
            yield {**config, "dropout": dropout}
'''
    (smanager_dir / "sweeps.py").write_text(sweeps_content, encoding="utf-8")


@cli.command()
@click.argument("script", type=click.Path(exists=True))
def info(script: str):
    """
    Show configuration info for a script.

    Displays the .smanager configuration that would be used for the given script.
    """
    try:
        config = SManagerConfig(Path(script))

        console.print(
            Panel(
                f"[bold]Script:[/bold] {script}",
                title="[bold blue]Configuration Info[/bold blue]",
                border_style="blue",
            )
        )

        if config.config_dir:
            console.print(
                f"[green]✓[/green] Found .smanager at: [cyan]{config.config_dir}[/cyan]"
            )

            if config.defaults:
                console.print("\n[bold]Defaults:[/bold]")
                for key, value in config.defaults.items():
                    console.print(f"  {key}: [green]{value}[/green]")
            else:
                console.print("\n[dim]No defaults configured[/dim]")

            if config.preamble:
                console.print("\n[bold]Preamble:[/bold]")
                console.print(
                    Panel(
                        Syntax(config.preamble, "bash", theme="monokai"),
                        border_style="dim",
                    )
                )
            else:
                console.print("\n[dim]No preamble configured[/dim]")
        else:
            console.print("[yellow]⚠[/yellow] No .smanager directory found")
            console.print("[dim]Run 'smanager init' to create one[/dim]")

    except OSError as exc:
        console.print(f"[red]✗ Error:[/red] {exc}")
        sys.exit(1)


@cli.command("list")
@click.argument("experiment", required=False)
def list_experiments(experiment: Optional[str]):
    """
    List generated sbatch scripts.

    \b
    EXPERIMENT: Experiment name to list scripts for (optional).
    """
    try:
        config = SManagerConfig()
        script_dir = config.get_script_dir()

        if not script_dir.exists():
            console.print("[yellow]⚠[/yellow] No scripts directory found")
            return

        if experiment:
            _list_experiment_scripts(script_dir, experiment)
        else:
            _list_all_experiments(script_dir)

    except OSError as exc:
        console.print(f"[red]✗ Error:[/red] {exc}")
        sys.exit(1)


def _list_experiment_scripts(script_dir: Path, experiment: str) -> None:
    """List scripts for a specific experiment."""
    exp_dir = script_dir / experiment
    if not exp_dir.exists():
        console.print(f"[yellow]⚠[/yellow] Experiment '{experiment}' not found")
        return

    console.print(f"[bold]Scripts for experiment: {experiment}[/bold]\n")
    for script_file in sorted(exp_dir.glob("**/*.sbatch")):
        console.print(f"  [cyan]{script_file.relative_to(script_dir)}[/cyan]")


def _list_all_experiments(script_dir: Path) -> None:
    """List all experiments."""
    experiments = [d for d in script_dir.iterdir() if d.is_dir()]

    if not experiments:
        console.print("[dim]No experiments found[/dim]")
        return

    table = Table(title="Experiments")
    table.add_column("Experiment", style="cyan")
    table.add_column("Scripts", style="green")

    for exp_dir in sorted(experiments):
        script_count = len(list(exp_dir.glob("**/*.sbatch")))
        table.add_row(exp_dir.name, str(script_count))

    console.print(table)
    console.print(f"\n[dim]Scripts directory: {script_dir}[/dim]")


def _find_latest_sweep(script_dir: Path) -> Optional[Tuple[Path, Dict]]:
    """Find the most recent sweep by looking at sweep.json files."""
    latest_sweep = None
    latest_time = None

    for sweep_json in script_dir.glob("**/sweep.json"):
        try:
            with open(sweep_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            created_at = data.get("created_at", "")
            if latest_time is None or created_at > latest_time:
                latest_time = created_at
                latest_sweep = (sweep_json.parent, data)
        except (json.JSONDecodeError, IOError):
            continue

    return latest_sweep


def _find_sweep_by_uuid(
    script_dir: Path, sweep_uuid: str
) -> Optional[Tuple[Path, Dict]]:
    """Find a sweep by its UUID."""
    for sweep_json in script_dir.glob("**/sweep.json"):
        try:
            with open(sweep_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("sweep_uuid") == sweep_uuid:
                return (sweep_json.parent, data)
            if sweep_json.parent.name.startswith(sweep_uuid):
                return (sweep_json.parent, data)
        except (json.JSONDecodeError, IOError):
            continue

    return None


@cli.command()
@click.argument("sweep_uuid", required=False)
@click.option(
    "--dry-run",
    "-d",
    is_flag=True,
    help="Show what would be cancelled without actually cancelling",
)
def kill(sweep_uuid: Optional[str], dry_run: bool):
    """
    Cancel all jobs from a sweep.

    \b
    SWEEP_UUID: UUID of the sweep to cancel (optional).
                If not provided, cancels jobs from the most recent sweep.

    \b
    Examples:
        smanager kill                    # Kill jobs from last sweep
        smanager kill --dry-run          # Show what would be killed
        smanager kill a1b2c3d4           # Kill specific sweep (partial UUID ok)
    """
    try:
        config = SManagerConfig()
        script_dir = config.get_script_dir()

        if not script_dir.exists():
            console.print("[yellow]⚠[/yellow] No scripts directory found")
            return

        result = _find_sweep_for_kill(script_dir, sweep_uuid)
        if result is None:
            return

        _, sweep_data = result
        _display_kill_info(sweep_data, dry_run)
        job_ids_to_cancel = _collect_jobs_to_cancel(sweep_data, dry_run)

        if not job_ids_to_cancel:
            console.print("\n[yellow]⚠[/yellow] No job IDs found to cancel")
            return

        _execute_kill(job_ids_to_cancel, dry_run)

    except OSError as exc:
        console.print(f"[red]✗ Error:[/red] {exc}")
        sys.exit(1)


def _find_sweep_for_kill(
    script_dir: Path, sweep_uuid: Optional[str]
) -> Optional[Tuple[Path, Dict]]:
    """Find the sweep to kill."""
    if sweep_uuid:
        result = _find_sweep_by_uuid(script_dir, sweep_uuid)
        if not result:
            console.print(f"[red]✗[/red] Sweep with UUID '{sweep_uuid}' not found")
            sys.exit(1)
        return result

    result = _find_latest_sweep(script_dir)
    if not result:
        console.print("[yellow]⚠[/yellow] No sweeps found")
        return None
    return result


def _display_kill_info(sweep_data: Dict, dry_run: bool) -> None:
    """Display information about the sweep to be killed."""
    title = (
        "[bold red]Kill Sweep Jobs[/bold red]"
        if not dry_run
        else "[bold yellow]Kill Sweep Jobs (Dry Run)[/bold yellow]"
    )
    console.print(
        Panel(
            f"[bold]Sweep UUID:[/bold] {sweep_data.get('sweep_uuid', 'N/A')}\n"
            f"[bold]Script:[/bold] {sweep_data.get('script', 'N/A')}\n"
            f"[bold]Created:[/bold] {sweep_data.get('created_at', 'N/A')}\n"
            f"[bold]Total Jobs:[/bold] {sweep_data.get('total_jobs', 0)}",
            title=title,
            border_style="red" if not dry_run else "yellow",
        )
    )


def _collect_jobs_to_cancel(sweep_data: Dict, dry_run: bool) -> List[str]:
    """Collect job IDs to cancel and display them."""
    jobs_data = sweep_data.get("jobs", {})
    job_ids_to_cancel = []

    table = Table(title="Jobs to Cancel")
    table.add_column("Job UUID", style="cyan")
    table.add_column("Index", style="dim")
    table.add_column("Slurm Job ID", style="green")
    table.add_column("Status", style="yellow")

    for job_uuid, job_info in jobs_data.items():
        slurm_job_id = job_info.get("slurm_job_id")
        if slurm_job_id:
            job_ids_to_cancel.append(slurm_job_id)
            status = "Will cancel" if not dry_run else "Would cancel"
            table.add_row(
                job_uuid[:8] + "...",
                str(job_info.get("index", "?")),
                slurm_job_id,
                status,
            )
        else:
            table.add_row(
                job_uuid[:8] + "...",
                str(job_info.get("index", "?")),
                "N/A",
                "[dim]No job ID (not submitted?)[/dim]",
            )

    console.print(table)
    return job_ids_to_cancel


def _execute_kill(job_ids_to_cancel: List[str], dry_run: bool) -> None:
    """Execute the kill command or show dry run info."""
    if dry_run:
        console.print(
            f"\n[yellow]⚠[/yellow] Dry run - would cancel {len(job_ids_to_cancel)} jobs"
        )
        console.print(f"[dim]  Command: scancel {' '.join(job_ids_to_cancel)}[/dim]")
        return

    console.print(f"\n[bold]Cancelling {len(job_ids_to_cancel)} jobs...[/bold]")

    try:
        result = subprocess.run(
            ["scancel"] + job_ids_to_cancel,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            console.print(
                f"[green]✓[/green] Successfully cancelled {len(job_ids_to_cancel)} jobs"
            )
        else:
            console.print("[yellow]⚠[/yellow] scancel completed with warnings")
            if result.stderr:
                console.print(f"[dim]{result.stderr.strip()}[/dim]")

    except FileNotFoundError:
        console.print("[red]✗[/red] scancel command not found. Is Slurm installed?")
        sys.exit(1)


@cli.command()
@click.option(
    "--last", "-l", "count", type=int, default=5, help="Show last N sweeps (default: 5)"
)
def history(count: int):
    """
    Show recent sweeps.

    Lists recent sweeps with their UUIDs, creation times, and job counts.
    """
    try:
        config = SManagerConfig()
        script_dir = config.get_script_dir()

        if not script_dir.exists():
            console.print("[yellow]⚠[/yellow] No scripts directory found")
            return

        sweeps = _collect_sweeps(script_dir)
        if not sweeps:
            console.print("[dim]No sweeps found[/dim]")
            return

        _display_sweep_history(sweeps[:count])

    except OSError as exc:
        console.print(f"[red]✗ Error:[/red] {exc}")
        sys.exit(1)


def _collect_sweeps(script_dir: Path) -> List[Dict]:
    """Collect all sweeps sorted by creation time."""
    sweeps = []
    for sweep_json in script_dir.glob("**/sweep.json"):
        try:
            with open(sweep_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            sweeps.append({"path": sweep_json.parent, "data": data})
        except (json.JSONDecodeError, IOError):
            continue

    sweeps.sort(key=lambda x: x["data"].get("created_at", ""), reverse=True)
    return sweeps


def _display_sweep_history(sweeps: List[Dict]) -> None:
    """Display sweep history table."""
    table = Table(title=f"Recent Sweeps (last {len(sweeps)})")
    table.add_column("UUID", style="cyan")
    table.add_column("Created", style="dim")
    table.add_column("Script", style="green")
    table.add_column("Function", style="yellow")
    table.add_column("Jobs", style="magenta")

    for sweep_item in sweeps:
        data = sweep_item["data"]
        uuid_short = data.get("sweep_uuid", "?")[:8]
        created = data.get("created_at", "?")[:19].replace("T", " ")
        script = Path(data.get("script", "?")).name
        func = data.get("sweep_function", "?")
        total = str(data.get("total_jobs", "?"))

        table.add_row(uuid_short, created, script, func, total)

    console.print(table)
    console.print("\n[dim]Use 'smanager kill <UUID>' to cancel jobs from a sweep[/dim]")


# Job status constants
JOB_STATUS_PENDING = {"PD", "CF"}  # Pending, Configuring
JOB_STATUS_RUNNING = {"R", "CG"}  # Running, Completing
JOB_STATUS_COMPLETED = {"CD"}  # Completed
JOB_STATUS_FAILED = {"F", "CA", "TO", "NF", "PR", "BF", "OOM"}  # Failed states


def _get_job_status(
    job_ids: List[str],
) -> Dict[str, Dict]:  # pylint: disable=too-many-branches
    """Get status of jobs using squeue and sacct."""
    if not job_ids:
        return {}

    status_map = {}

    # First try squeue for running/pending jobs
    try:
        result = subprocess.run(
            ["squeue", "-j", ",".join(job_ids), "-h", "-o", "%A|%t|%M|%N|%j"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split("|")
                    if len(parts) >= 5:
                        job_id, state, time_used, node, name = parts[:5]
                        status_map[job_id] = {
                            "state": state,
                            "time": time_used,
                            "node": node or "-",
                            "name": name,
                            "source": "squeue",
                        }
    except FileNotFoundError:
        pass

    # Then check sacct for completed/failed jobs not in squeue
    missing_jobs = [j for j in job_ids if j not in status_map]
    if missing_jobs:
        _get_sacct_status(missing_jobs, status_map)

    return status_map


def _get_sacct_status(missing_jobs: List[str], status_map: Dict[str, Dict]) -> None:
    """Get job status from sacct for completed/failed jobs."""
    try:
        result = subprocess.run(
            [
                "sacct",
                "-j",
                ",".join(missing_jobs),
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
        return

    if result.returncode != 0:
        return

    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 5:
            continue
        job_id, state, elapsed, node, exit_code = parts[:5]
        # Skip job steps (like 12345.batch)
        if "." in job_id or job_id not in missing_jobs:
            continue
        status_map[job_id] = {
            "state": state,
            "time": elapsed,
            "node": node or "-",
            "exit_code": exit_code,
            "source": "sacct",
        }


def _read_error_tail(error_file: str, lines: int = 20) -> Optional[str]:
    """Read the last N lines of an error file."""
    error_path = Path(error_file)
    if not error_path.exists():
        return None

    try:
        content = error_path.read_text(encoding="utf-8", errors="replace")
        error_lines = content.strip().split("\n")
        if error_lines and any(line.strip() for line in error_lines):
            return "\n".join(error_lines[-lines:])
    except OSError:
        pass

    return None


def _format_state(state: str) -> str:
    """Format job state with color."""
    if state in JOB_STATUS_PENDING:
        return f"[yellow]{state}[/yellow]"
    if state in JOB_STATUS_RUNNING:
        return f"[blue]{state}[/blue]"
    if state in JOB_STATUS_COMPLETED:
        return f"[green]{state}[/green]"
    if state in JOB_STATUS_FAILED:
        return f"[red]{state}[/red]"
    return state


def _build_monitor_table(
    sweep_data: Dict, status_map: Dict[str, Dict], show_params: bool = True
) -> Table:
    """Build the monitoring table."""
    table = Table(title="Job Status", show_header=True)
    table.add_column("Job ID", style="cyan")
    table.add_column("Status")
    table.add_column("Time", style="dim")
    table.add_column("Node", style="dim")
    if show_params:
        table.add_column("Parameters", style="yellow")

    jobs_data = sweep_data.get("jobs", {})

    for job_info in jobs_data.values():
        slurm_job_id = job_info.get("slurm_job_id")
        if not slurm_job_id:
            continue

        status = status_map.get(slurm_job_id, {})
        state = status.get("state", "?")
        time_used = status.get("time", "-")
        node = status.get("node", "-")

        if show_params:
            params = job_info.get("params", {})
            params_str = ", ".join(f"{k}={v}" for k, v in params.items())
            if len(params_str) > 40:
                params_str = params_str[:37] + "..."
            table.add_row(
                slurm_job_id, _format_state(state), time_used, node, params_str
            )
        else:
            table.add_row(slurm_job_id, _format_state(state), time_used, node)

    return table


@cli.command()
@click.argument("sweep_uuid", required=False)
@click.option(
    "--interval",
    "-i",
    type=float,
    default=10.0,
    help="Polling interval in seconds (default: 10)",
)
@click.option(
    "--timeout",
    "-t",
    type=float,
    default=0,
    help="Timeout in seconds (0 = no timeout, default: 0)",
)
@click.option(
    "--show-errors",
    "-e",
    is_flag=True,
    help="Show error file contents for failed jobs",
)
@click.option(
    "--error-lines",
    type=int,
    default=20,
    help="Number of error lines to show (default: 20)",
)
def monitor(
    sweep_uuid: Optional[str],
    interval: float,
    timeout: float,
    show_errors: bool,
    error_lines: int,
):
    """
    Monitor jobs from a sweep until completion.

    \b
    SWEEP_UUID: UUID of the sweep to monitor (optional).
                If not provided, monitors the most recent sweep.

    \b
    Press Ctrl+C to stop monitoring.

    \b
    Examples:
        smanager monitor                    # Monitor last sweep
        smanager monitor --interval 5       # Poll every 5 seconds
        smanager monitor --timeout 3600     # Stop after 1 hour
        smanager monitor -e                 # Show errors for failed jobs
        smanager monitor a1b2c3d4           # Monitor specific sweep
    """
    try:
        config = SManagerConfig()
        script_dir = config.get_script_dir()

        if not script_dir.exists():
            console.print("[yellow]⚠[/yellow] No scripts directory found")
            return

        # Find the sweep
        result = _find_sweep_for_kill(script_dir, sweep_uuid)
        if result is None:
            return

        _, sweep_data = result
        jobs_data = sweep_data.get("jobs", {})

        # Get list of job IDs
        job_ids = [
            info.get("slurm_job_id")
            for info in jobs_data.values()
            if info.get("slurm_job_id")
        ]

        if not job_ids:
            console.print("[yellow]⚠[/yellow] No submitted jobs found to monitor")
            return

        console.print(
            Panel(
                f"[bold]Sweep UUID:[/bold] {sweep_data.get('sweep_uuid', 'N/A')}\n"
                f"[bold]Script:[/bold] {sweep_data.get('script', 'N/A')}\n"
                f"[bold]Jobs:[/bold] {len(job_ids)}\n"
                f"[bold]Interval:[/bold] {interval}s"
                + (f"\n[bold]Timeout:[/bold] {timeout}s" if timeout > 0 else ""),
                title="[bold blue]Monitoring Jobs[/bold blue]",
                border_style="blue",
            )
        )
        console.print("\n[dim]Press Ctrl+C to stop monitoring[/dim]\n")

        _run_monitor_loop(
            sweep_data, job_ids, interval, timeout, show_errors, error_lines
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped by user[/yellow]")
    except OSError as exc:
        console.print(f"[red]✗ Error:[/red] {exc}")
        sys.exit(1)


def _run_monitor_loop(  # pylint: disable=too-many-positional-arguments
    sweep_data: Dict,
    job_ids: List[str],
    interval: float,
    timeout: float,
    show_errors: bool,
    error_lines: int,
) -> None:
    """Run the monitoring loop."""
    start_time = time.time()
    shown_errors: Set[str] = set()
    all_complete = False

    with Live(console=console, refresh_per_second=1) as live:
        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if 0 < timeout <= elapsed:
                console.print(f"\n[yellow]⚠[/yellow] Timeout reached ({timeout}s)")
                break

            # Get job status
            status_map = _get_job_status(job_ids)

            # Build and display table
            table = _build_monitor_table(sweep_data, status_map)
            live.update(table)

            # Check for completion and failures
            completed = 0
            failed = 0
            failed_jobs = []

            for job_id in job_ids:
                status = status_map.get(job_id, {})
                state = status.get("state", "")

                if state in JOB_STATUS_COMPLETED:
                    completed += 1
                elif state in JOB_STATUS_FAILED:
                    failed += 1
                    failed_jobs.append(job_id)

            # Show errors for newly failed jobs
            if show_errors and failed_jobs:
                _show_job_errors(sweep_data, failed_jobs, shown_errors, error_lines)

            # Check if all done
            if completed + failed >= len(job_ids):
                all_complete = True
                break

            time.sleep(interval)

    # Final summary
    console.print("\n")
    _print_monitor_summary(job_ids, _get_job_status(job_ids), all_complete)


def _show_job_errors(
    sweep_data: Dict,
    failed_jobs: List[str],
    shown_errors: Set[str],
    error_lines: int,
) -> None:
    """Show error output for failed jobs."""
    jobs_data = sweep_data.get("jobs", {})

    for job_info in jobs_data.values():
        slurm_job_id = job_info.get("slurm_job_id")
        if slurm_job_id not in failed_jobs:
            continue
        if slurm_job_id in shown_errors:
            continue

        # Look for error file in the sweep directory logs
        # We stored the job_uuid in the sweep.json, error files are named {job_uuid}.err
        error_file = None
        for key in jobs_data:
            if jobs_data[key].get("slurm_job_id") == slurm_job_id:
                # Construct error path based on job UUID
                sweep_dir = Path(sweep_data.get("script", ".")).parent
                # Try common locations
                possible_paths = [
                    sweep_dir / "logs" / f"{key}.err",
                    Path(f"slurm-{slurm_job_id}.out"),
                ]
                for p in possible_paths:
                    if p.exists():
                        error_file = str(p)
                        break
                break

        if error_file:
            error_content = _read_error_tail(error_file, error_lines)
            if error_content:
                console.print(
                    Panel(
                        Syntax(error_content, "text", theme="monokai"),
                        title=f"[bold red]Error: Job {slurm_job_id}[/bold red]",
                        border_style="red",
                    )
                )
                shown_errors.add(slurm_job_id)


def _print_monitor_summary(
    job_ids: List[str], status_map: Dict[str, Dict], all_complete: bool
) -> None:
    """Print final monitoring summary."""
    completed = 0
    failed = 0
    other = 0

    for job_id in job_ids:
        status = status_map.get(job_id, {})
        state = status.get("state", "")

        if state in JOB_STATUS_COMPLETED:
            completed += 1
        elif state in JOB_STATUS_FAILED:
            failed += 1
        else:
            other += 1

    total = len(job_ids)

    if all_complete:
        if failed == 0:
            console.print(
                Panel(
                    f"[bold green]All {total} jobs completed![/bold green]",
                    border_style="green",
                )
            )
        else:
            console.print(
                Panel(
                    f"[bold]Completed:[/bold] {completed}/{total}\n"
                    f"[bold red]Failed:[/bold red] {failed}/{total}",
                    title="[bold yellow]Jobs Finished[/bold yellow]",
                    border_style="yellow",
                )
            )
    else:
        console.print(
            Panel(
                f"[bold]Completed:[/bold] {completed}/{total}\n"
                f"[bold]Failed:[/bold] {failed}/{total}\n"
                f"[bold]Other:[/bold] {other}/{total}",
                title="[bold blue]Monitoring Summary[/bold blue]",
                border_style="blue",
            )
        )


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
