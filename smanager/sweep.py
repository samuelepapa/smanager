"""Parameter sweep handling for running multiple jobs."""

import importlib.util
import json
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator, Callable
from datetime import datetime

from .config import SManagerConfig
from .job import SlurmJob


def load_sweep_function(file_path: str, function_name: str) -> Callable[[], Generator[Dict[str, Any], None, None]]:
    """
    Load a sweep generator function from a Python file.
    
    Args:
        file_path: Path to the Python file containing the sweep function.
        function_name: Name of the function to load.
    
    Returns:
        The sweep generator function.
    
    Raises:
        FileNotFoundError: If the file doesn't exist.
        AttributeError: If the function doesn't exist in the file.
        TypeError: If the function is not callable.
    """
    file_path = Path(file_path).resolve()
    
    if not file_path.exists():
        raise FileNotFoundError(f"Sweep file not found: {file_path}")
    
    # Load the module from file
    spec = importlib.util.spec_from_file_location("sweep_module", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Get the function
    if not hasattr(module, function_name):
        raise AttributeError(f"Function '{function_name}' not found in {file_path}")
    
    func = getattr(module, function_name)
    
    if not callable(func):
        raise TypeError(f"'{function_name}' is not callable")
    
    return func


class Sweep:
    """Manages a parameter sweep across multiple jobs."""
    
    def __init__(
        self,
        script_path: str,
        sweep_file: str,
        sweep_function: str,
        base_args: Optional[List[str]] = None,
        experiment_name: Optional[str] = None,
        config: Optional[SManagerConfig] = None,
        arg_format: str = "--{key}={value}",
        # Slurm options
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
        working_dir: Optional[str] = None,
    ):
        """
        Initialize a parameter sweep.
        
        Args:
            script_path: Path to the Python script to run.
            sweep_file: Path to Python file containing the sweep generator function.
            sweep_function: Name of the generator function in sweep_file.
            base_args: Base arguments always passed to the script.
            experiment_name: Name for the experiment (for folder organization only).
            config: SManagerConfig for project settings.
            arg_format: Format string for arguments (default: "--{key}={value}").
            ... (Slurm options passed to all jobs)
        """
        self.script_path = Path(script_path).resolve()
        self.sweep_file = Path(sweep_file).resolve()
        self.sweep_function_name = sweep_function
        self.base_args = base_args or []
        self.arg_format = arg_format
        
        # Load the sweep generator function
        self.sweep_function = load_sweep_function(str(self.sweep_file), self.sweep_function_name)
        
        # Use parent.script as experiment name if not provided
        # e.g., "fitting/train.py" -> "fitting.train"
        if experiment_name:
            self.experiment_name = experiment_name
        else:
            parent_name = self.script_path.parent.name
            script_name = self.script_path.stem
            self.experiment_name = f"{parent_name}.{script_name}"
        
        self.config = config or SManagerConfig(self.script_path)
        
        # Store Slurm options
        self.slurm_options = {
            'partition': partition,
            'gpus': gpus,
            'memory': memory,
            'time': time,
            'nodes': nodes,
            'ntasks': ntasks,
            'cpus_per_task': cpus_per_task,
            'output': output,
            'error': error,
            'mail_type': mail_type,
            'mail_user': mail_user,
            'account': account,
            'qos': qos,
            'constraint': constraint,
            'exclude': exclude,
            'nodelist': nodelist,
            'extra_sbatch_args': extra_sbatch_args,
            'python_executable': python_executable,
        }
        self.working_dir = working_dir
        
        # Generate UUID for the entire sweep
        self.sweep_uuid = str(uuid.uuid4())
        
        # Track generated jobs
        self.jobs: List[SlurmJob] = []
        self.sweep_dir: Optional[Path] = None
    
    def _format_args(self, params: Dict[str, Any]) -> List[str]:
        """Format parameter dict into command line arguments."""
        args = []
        for key, value in params.items():
            arg = self.arg_format.format(key=key, value=value)
            args.append(arg)
        return args
    
    def generate_jobs(self) -> List[SlurmJob]:
        """
        Generate all jobs for the sweep by calling the sweep generator.
        
        Returns:
            List of configured SlurmJob instances.
        """
        self.jobs = []
        
        # Call the generator function and iterate over all parameter sets
        generator = self.sweep_function()
        
        for idx, params in enumerate(generator):
            if not isinstance(params, dict):
                raise TypeError(
                    f"Sweep generator must yield dictionaries, got {type(params).__name__} at index {idx}"
                )
            
            # Combine base args with sweep params
            sweep_args = self._format_args(params)
            all_args = self.base_args + sweep_args
            
            job = SlurmJob(
                script_path=str(self.script_path),
                script_args=all_args,
                experiment_name=self.experiment_name,
                config=self.config,
                working_dir=self.working_dir,
                **self.slurm_options
            )
            
            # Store params on job for reference
            job.sweep_params = params
            job.sweep_index = idx
            
            self.jobs.append(job)
        
        return self.jobs
    
    def save_scripts(self, script_dir: Optional[Path] = None) -> List[Path]:
        """
        Save all job scripts to disk.
        
        Args:
            script_dir: Base directory for scripts.
        
        Returns:
            List of paths to saved script files.
        """
        if not self.jobs:
            self.generate_jobs()
        
        if script_dir is None:
            script_dir = self.config.get_script_dir()
        
        # Create sweep directory using UUID
        self.sweep_dir = script_dir / self.experiment_name / self.sweep_uuid
        self.sweep_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logs subdirectory
        logs_dir = self.sweep_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        paths = []
        for job in self.jobs:
            # Set default log paths if not specified
            if job.output is None:
                job.output = str(logs_dir / f"{job.job_uuid}.out")
            if job.error is None:
                job.error = str(logs_dir / f"{job.job_uuid}.err")
            
            # Each job uses its own UUID as the job name and filename
            script_content = job.generate_script()
            script_path = self.sweep_dir / f"{job.job_uuid}.sbatch"
            script_path.write_text(script_content)
            job.sbatch_script_path = script_path
            
            paths.append(script_path)
        
        # Save sweep mapping as JSON
        self._save_sweep_mapping()
        
        return paths
    
    def _save_sweep_mapping(self) -> None:
        """Save a JSON file mapping job UUIDs to their parameters."""
        if not self.sweep_dir:
            return
        
        mapping_path = self.sweep_dir / "sweep.json"
        
        # Convert params to JSON-serializable format
        def make_serializable(obj):
            """Convert non-serializable objects to strings."""
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {str(k): make_serializable(v) for k, v in obj.items()}
            else:
                return str(obj)
        
        mapping = {
            "sweep_uuid": self.sweep_uuid,
            "script": str(self.script_path),
            "sweep_file": str(self.sweep_file),
            "sweep_function": self.sweep_function_name,
            "base_args": self.base_args,
            "arg_format": self.arg_format,
            "total_jobs": len(self.jobs),
            "created_at": datetime.now().isoformat(),
            "jobs": {}
        }
        
        for job in self.jobs:
            mapping["jobs"][job.job_uuid] = {
                "index": job.sweep_index,
                "params": make_serializable(job.sweep_params),
                "slurm_job_id": job.job_id,
            }
        
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)
    
    def submit_all(self, dry_run: bool = False, delay: float = 0.0) -> List[Optional[str]]:
        """
        Submit all jobs to Slurm.
        
        Args:
            dry_run: If True, save scripts but don't submit.
            delay: Seconds to wait between submissions.
        
        Returns:
            List of job IDs (or None for dry run).
        """
        import time
        
        if not self.jobs or not self.jobs[0].sbatch_script_path:
            self.save_scripts()
        
        job_ids = []
        for job in self.jobs:
            job_id = job.submit(dry_run=dry_run)
            job_ids.append(job_id)
            
            if delay > 0 and not dry_run:
                time.sleep(delay)
        
        # Update mapping with job IDs after submission
        if not dry_run:
            self._save_sweep_mapping()
        
        return job_ids
    
    def __len__(self) -> int:
        """Return the number of jobs in the sweep."""
        if self.jobs:
            return len(self.jobs)
        # Generate jobs to count them
        self.generate_jobs()
        return len(self.jobs)
