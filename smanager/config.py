"""Configuration handling and .smanager directory discovery."""

from pathlib import Path
from typing import Optional

import yaml


class SManagerConfig:
    """Handles configuration discovery and loading from .smanager directories."""

    CONFIG_DIR_NAME = ".smanager"
    PREAMBLE_FILE = "preamble.sh"
    CONFIG_FILE = "config.yaml"

    def __init__(self, script_path: Optional[Path] = None):
        """
        Initialize configuration by searching for .smanager directory.

        Args:
            script_path: Path to the target Python script. If provided,
                        searches upward from this location for .smanager.
        """
        self.script_path = Path(script_path).resolve() if script_path else None
        self.config_dir: Optional[Path] = None
        self.preamble: str = ""
        self.defaults: dict = {}

        if self.script_path:
            self._discover_config()

    def _discover_config(self) -> None:
        """Search upward from script location to find .smanager directory."""
        if not self.script_path:
            return

        search_path = (
            self.script_path.parent if self.script_path.is_file() else self.script_path
        )

        while search_path != search_path.parent:  # Stop at filesystem root
            potential_config = search_path / self.CONFIG_DIR_NAME
            if potential_config.is_dir():
                self.config_dir = potential_config
                self._load_config()
                return
            search_path = search_path.parent

    def _load_config(self) -> None:
        """Load configuration from discovered .smanager directory."""
        if not self.config_dir:
            return

        # Load preamble
        preamble_path = self.config_dir / self.PREAMBLE_FILE
        if preamble_path.exists():
            self.preamble = preamble_path.read_text(encoding="utf-8")

        # Load config defaults
        config_path = self.config_dir / self.CONFIG_FILE
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.defaults = yaml.safe_load(f) or {}

    def get_preamble(self) -> str:
        """Return the preamble script content."""
        return self.preamble

    def get_default(self, key: str, fallback=None):
        """Get a default value from config."""
        return self.defaults.get(key, fallback)

    def get_script_dir(self) -> Path:
        """Get the SCRIPT_DIR path for storing generated scripts."""
        if self.config_dir:
            script_dir = self.config_dir / "scripts"
        else:
            script_dir = Path.cwd() / ".smanager" / "scripts"

        script_dir.mkdir(parents=True, exist_ok=True)
        return script_dir


def find_project_root(start_path: Path) -> Optional[Path]:
    """Find the project root by looking for .smanager directory."""
    search_path = start_path.resolve()
    if search_path.is_file():
        search_path = search_path.parent

    while search_path != search_path.parent:
        if (search_path / SManagerConfig.CONFIG_DIR_NAME).is_dir():
            return search_path
        search_path = search_path.parent

    return None
