"""Sbatch script templates and generation."""

from typing import Optional, List, Dict, Any
from jinja2 import Template


# Default sbatch template
DEFAULT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={{ job_name }}
#SBATCH --output={{ output }}
#SBATCH --error={{ error }}
{%- if partition %}
#SBATCH --partition={{ partition }}
{%- endif %}
{%- if gpus %}
#SBATCH --gres=gpu:{{ gpus }}
{%- endif %}
{%- if memory %}
#SBATCH --mem={{ memory }}
{%- endif %}
{%- if time %}
#SBATCH --time={{ time }}
{%- endif %}
{%- if nodes %}
#SBATCH --nodes={{ nodes }}
{%- endif %}
{%- if ntasks %}
#SBATCH --ntasks={{ ntasks }}
{%- endif %}
{%- if cpus_per_task %}
#SBATCH --cpus-per-task={{ cpus_per_task }}
{%- endif %}
{%- if mail_type %}
#SBATCH --mail-type={{ mail_type }}
{%- endif %}
{%- if mail_user %}
#SBATCH --mail-user={{ mail_user }}
{%- endif %}
{%- if account %}
#SBATCH --account={{ account }}
{%- endif %}
{%- if qos %}
#SBATCH --qos={{ qos }}
{%- endif %}
{%- if constraint %}
#SBATCH --constraint={{ constraint }}
{%- endif %}
{%- if exclude %}
#SBATCH --exclude={{ exclude }}
{%- endif %}
{%- if nodelist %}
#SBATCH --nodelist={{ nodelist }}
{%- endif %}
{%- if extra_sbatch_args %}
{% for arg in extra_sbatch_args %}
#SBATCH {{ arg }}
{%- endfor %}
{%- endif %}

# Preamble (from .smanager/preamble.sh if found)
{% if preamble %}
{{ preamble }}
{% endif %}

# Change to working directory
cd {{ working_dir }}

# Run the Python script
{{ python_executable }} {{ script_path }}{% if script_args %} {{ script_args }}{% endif %}
"""


class SbatchTemplate:
    """Generates sbatch scripts from templates."""
    
    def __init__(self, template_str: Optional[str] = None):
        """
        Initialize with a template string.
        
        Args:
            template_str: Custom Jinja2 template string. Uses default if None.
        """
        self.template_str = template_str or DEFAULT_TEMPLATE
        self.template = Template(self.template_str)
    
    def render(
        self,
        script_path: str,
        job_name: str,
        working_dir: str,
        script_args: str = "",
        preamble: str = "",
        python_executable: str = "python",
        # Slurm options (all optional)
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
        **kwargs
    ) -> str:
        """
        Render the sbatch script with given parameters.
        
        Only parameters that are explicitly provided (not None) will be
        included in the generated script.
        
        Returns:
            The rendered sbatch script as a string.
        """
        context = {
            'script_path': script_path,
            'job_name': job_name,
            'working_dir': working_dir,
            'script_args': script_args,
            'preamble': preamble.strip() if preamble else "",
            'python_executable': python_executable,
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
            'extra_sbatch_args': extra_sbatch_args or [],
        }
        context.update(kwargs)
        
        return self.template.render(**context)


def load_template_from_file(template_path: str) -> SbatchTemplate:
    """Load a custom template from a file."""
    with open(template_path, 'r') as f:
        template_str = f.read()
    return SbatchTemplate(template_str)

