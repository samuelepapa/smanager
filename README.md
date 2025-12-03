# Slurm Manager (`smanager`)

[![CI](https://github.com/samuelepapa/smanager/actions/workflows/ci.yml/badge.svg)](https://github.com/samuelepapa/smanager/actions/workflows/ci.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A Python CLI tool for managing Slurm jobs and parameter sweeps with project-level configuration support.

## Features

- **Simple Job Submission**: Run Python scripts as Slurm jobs with intuitive CLI options
- **Generator-based Sweeps**: Define sweeps as Python generator functions for maximum flexibility
- **Local Sweeps**: Run sweeps locally on an allocated node using tmux sessions with GPU assignment
- **Project Configuration**: Define project-level defaults and environment setup via `.smanager` directory
- **Script Organization**: Automatically organizes generated sbatch scripts by experiment name
- **Preamble Support**: Add common setup commands (conda activation, environment variables, etc.) that apply to all jobs in a project
- **Job Management**: Kill sweeps, view history, and manage running jobs

## Installation

```bash
# Install in development mode
pip install -e .

# Or install directly
pip install .
```

## Quick Start

### 1. Initialize a Project

```bash
cd your_project/
smanager init
```

This creates a `.smanager` directory with:
- `config.yaml` - Default Slurm options
- `preamble.sh` - Environment setup commands
- `sweeps.py` - Example sweep generator functions
- `scripts/` - Directory for generated sbatch scripts

### 2. Configure Your Project

Edit `.smanager/preamble.sh` to set up your environment:

```bash
# Activate conda environment
source ~/miniconda3/bin/activate
conda activate myenv

# Set environment variables
export DATA_PATH=/path/to/data
export WANDB_PROJECT=my_project
```

Edit `.smanager/config.yaml` to set default Slurm options:

```yaml
partition: gpu
gpus: 1
memory: 32G
time: "24:00:00"
cpus_per_task: 4
```

### 3. Run a Job

```bash
# Simple job
smanager run train.py --gpus 4 --memory 64G

# With script arguments (after --)
smanager run train.py --gpus 2 -- --lr 0.01 --epochs 100

# Dry run (generate script without submitting)
smanager run train.py --gpus 4 --dry-run --show
```

### 4. Run a Parameter Sweep

Create a sweep file with generator functions:

```python
# sweeps.py
def learning_rate_sweep():
    """Sweep over learning rates."""
    for lr in [0.1, 0.01, 0.001]:
        yield {"lr": lr}

def grid_search():
    """Grid search over multiple parameters."""
    for lr in [0.1, 0.01, 0.001]:
        for batch_size in [32, 64, 128]:
            yield {
                "lr": lr,
                "batch_size": batch_size,
            }

def custom_sweep():
    """Complex sweep with conditional logic."""
    configs = [
        {"model": "small", "hidden": 256},
        {"model": "large", "hidden": 1024},
    ]
    for config in configs:
        for dropout in [0.1, 0.3]:
            yield {**config, "dropout": dropout}
```

Run the sweep:

```bash
# Basic sweep
smanager sweep train.py sweeps.py learning_rate_sweep --gpus 4

# Grid search with slurm options
smanager sweep train.py sweeps.py grid_search -g 2 -t 12:00:00

# Dry run - generate scripts without submitting
smanager sweep train.py sweeps.py grid_search --dry-run

# With base arguments passed to all jobs
smanager sweep train.py sweeps.py custom_sweep -- --epochs 100 --data cifar10
```

### 5. Manage Jobs

```bash
# View recent sweeps
smanager history

# Kill all jobs from the last sweep
smanager kill

# Kill jobs from a specific sweep (partial UUID works)
smanager kill a1b2c3d4

# Dry run - see what would be killed
smanager kill --dry-run
```

### 6. Run Local Sweeps (on an Allocated Node)

When you have an allocated node with multiple GPUs (e.g., via `salloc`), you can run a sweep locally using tmux sessions instead of submitting multiple Slurm jobs:

```bash
# Run sweep with 4 workers, each assigned one GPU
smanager local train.py sweeps.py grid_search --workers 4 --gpus 0,1,2,3

# Dry run to preview generated scripts
smanager local train.py sweeps.py my_sweep -w 4 -g 0,1,2,3 --dry-run --show

# With base arguments
smanager local train.py sweeps.py my_sweep -w 4 -g 0,1,2,3 -- --epochs 100

# List active sweep sessions
smanager local-list

# Kill all sweep sessions
smanager local-kill
```

Each worker runs in its own tmux session with:
- Assigned GPU(s) via `CUDA_VISIBLE_DEVICES`
- Environment setup from your preamble
- A subset of the sweep jobs running sequentially

## CLI Reference

### `smanager run`

Run a Python script as a Slurm job.

```
smanager run [OPTIONS] SCRIPT [SCRIPT_ARGS]...
```

**Options:**
| Option | Short | Description |
|--------|-------|-------------|
| `--name` | `-n` | Experiment name (defaults to `parent.script`) |
| `--gpus` | `-g` | Number of GPUs |
| `--memory` | `-m` | Memory (e.g., 32G, 64000M) |
| `--time` | `-t` | Time limit (e.g., 24:00:00) |
| `--partition` | `-p` | Slurm partition |
| `--nodes` | | Number of nodes |
| `--ntasks` | | Number of tasks |
| `--cpus-per-task` | `-c` | CPUs per task |
| `--account` | `-A` | Account for billing |
| `--qos` | | Quality of Service |
| `--constraint` | | Node feature constraint |
| `--exclude` | | Nodes to exclude |
| `--nodelist` | | Specific nodes to use |
| `--output` | `-o` | Output file pattern |
| `--error` | `-e` | Error file pattern |
| `--mail-type` | | Mail notification type |
| `--mail-user` | | Email for notifications |
| `--python` | | Python executable (default: python) |
| `--working-dir` | `-w` | Working directory (default: current directory) |
| `--dry-run` | `-d` | Generate script without submitting |
| `--show` | `-s` | Display generated script |
| `--sbatch-arg` | | Extra sbatch arguments (repeatable) |

### `smanager sweep`

Run a parameter sweep with multiple jobs using a generator function.

```
smanager sweep [OPTIONS] SCRIPT SWEEP_FILE SWEEP_FUNCTION [BASE_ARGS]...
```

**Arguments:**
- `SCRIPT`: Path to the Python script to run
- `SWEEP_FILE`: Python file containing the sweep generator function
- `SWEEP_FUNCTION`: Name of the generator function that yields parameter dicts
- `BASE_ARGS`: Base arguments always passed to the script (after --)

**Additional Options:**
| Option | Description |
|--------|-------------|
| `--dry-run` / `-d` | Generate scripts without submitting |
| `--delay` | Delay between submissions (seconds) |
| `--arg-format` | Argument format (default: `--{key}={value}`) |

### `smanager kill`

Cancel all jobs from a sweep.

```
smanager kill [SWEEP_UUID] [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--dry-run` / `-d` | Show what would be cancelled |
| `--last` / `-l` | Kill jobs from the last sweep (default) |

### `smanager history`

Show recent sweeps.

```
smanager history [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--last` / `-l` | Number of sweeps to show (default: 5) |

### `smanager init`

Initialize a `.smanager` directory.

```
smanager init [PATH]
```

### `smanager info`

Show configuration info for a script.

```
smanager info SCRIPT
```

### `smanager list`

List generated sbatch scripts.

```
smanager list [EXPERIMENT]
```

### `smanager local`

Run a parameter sweep locally using tmux sessions. Useful when you have an allocated node with multiple GPUs.

```
smanager local [OPTIONS] SCRIPT SWEEP_FILE SWEEP_FUNCTION [BASE_ARGS]...
```

**Arguments:**
- `SCRIPT`: Path to the Python script to run
- `SWEEP_FILE`: Python file containing the sweep generator function
- `SWEEP_FUNCTION`: Name of the generator function that yields parameter dicts
- `BASE_ARGS`: Base arguments always passed to the script (after --)

**Options:**
| Option | Short | Description |
|--------|-------|-------------|
| `--workers` | `-w` | Number of parallel tmux sessions/workers |
| `--gpus` | `-g` | Comma-separated GPU IDs (e.g., '0,1,2,3') |
| `--name` | `-n` | Experiment name |
| `--session-prefix` | `-s` | Prefix for tmux session names (default: sweep) |
| `--python` | | Python executable (default: python) |
| `--working-dir` | | Working directory |
| `--dry-run` | `-d` | Generate scripts without launching tmux |
| `--show` | | Display generated worker scripts |
| `--arg-format` | | Argument format (default: `--{key}={value}`) |

**Examples:**
```bash
# 4 workers with one GPU each
smanager local train.py sweeps.py grid --workers 4 --gpus 0,1,2,3

# Custom session prefix
smanager local train.py sweeps.py grid -w 2 -g 0,1 --session-prefix myexp
```

### `smanager local-kill`

Kill tmux sessions created by local sweep.

```
smanager local-kill [PREFIX] [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--dry-run` / `-d` | Show what would be killed |

### `smanager local-list`

List active tmux sessions from local sweeps.

```
smanager local-list [PREFIX]
```

## Project Structure

When you run jobs, scripts and logs are organized by UUID. Folder names use `parent.script` format based on the script location:

```
your_project/
├── .smanager/
│   ├── config.yaml
│   ├── preamble.sh
│   ├── sweeps.py
│   └── scripts/
│       ├── experiments.train/                  # from experiments/train.py
│       │   ├── a1b2c3d4-...-ef12.sbatch       # single job (UUID)
│       │   └── logs/
│       │       ├── a1b2c3d4-...-ef12.out      # stdout
│       │       └── a1b2c3d4-...-ef12.err      # stderr
│       └── experiments.train/                  # sweeps use same folder
│           └── f5e6d7c8-...-1234/             # sweep UUID
│               ├── sweep.json                  # parameter mapping
│               ├── abc123-....sbatch          # job 0
│               ├── def456-....sbatch          # job 1
│               └── logs/
│                   ├── abc123-....out
│                   ├── abc123-....err
│                   └── ...
├── experiments/
│   └── train.py
└── sweeps.py
```

**Naming examples:**
| Script Path | Folder Name |
|-------------|-------------|
| `train.py` | `project.train` |
| `experiments/train.py` | `experiments.train` |
| `src/models/vae.py` | `models.vae` |

The `sweep.json` file maps job UUIDs to their parameters:

```json
{
  "sweep_uuid": "f5e6d7c8-...",
  "script": "/path/to/train.py",
  "sweep_function": "grid_search",
  "total_jobs": 9,
  "jobs": {
    "abc123-...": {
      "index": 0,
      "params": {"lr": 0.1, "batch_size": 32},
      "slurm_job_id": "12345"
    }
  }
}
```

## Configuration Discovery

`smanager` searches for `.smanager` directory starting from the current directory and moving up to parent directories. This allows you to:

- Have project-level configuration in the project root
- Override with more specific configuration closer to scripts
- Share configuration across multiple scripts in a project

## Sweep Examples

### Simple Grid Search

```python
def hyperparameter_grid():
    learning_rates = [1e-4, 1e-3, 1e-2]
    weight_decays = [0, 1e-4, 1e-3]

    for lr in learning_rates:
        for wd in weight_decays:
            yield {"learning_rate": lr, "weight_decay": wd}
```

### Conditional Sweeps

```python
def architecture_sweep():
    architectures = {
        "small": {"layers": 2, "hidden": 256},
        "medium": {"layers": 4, "hidden": 512},
        "large": {"layers": 6, "hidden": 1024},
    }

    for name, config in architectures.items():
        # Small models get less regularization
        dropout = 0.1 if name == "small" else 0.3
        yield {
            "arch": name,
            "num_layers": config["layers"],
            "hidden_dim": config["hidden"],
            "dropout": dropout,
        }
```

### Reading from Files

```python
import json

def configs_from_file():
    with open("experiments.json") as f:
        experiments = json.load(f)

    for exp in experiments:
        yield exp
```

### Random Search

```python
import random

def random_search(n_trials=50):
    random.seed(42)

    for _ in range(n_trials):
        yield {
            "lr": 10 ** random.uniform(-5, -2),
            "dropout": random.uniform(0.1, 0.5),
            "hidden": random.choice([256, 512, 1024]),
        }
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/samuelepapa/smanager.git
cd smanager

# Install with dev dependencies
make install-dev

# This also installs pre-commit hooks
```

### Code Quality

We use the following tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **Pylint**: Linting with Google style guidelines

```bash
# Format code
make format

# Run linters
make lint

# Run tests
make test
```

### Pre-commit Hooks

Pre-commit hooks run automatically on each commit:

```bash
# Install hooks (done automatically by make install-dev)
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage report
pytest tests/ -v --cov=smanager --cov-report=html
```

## License

MIT License
