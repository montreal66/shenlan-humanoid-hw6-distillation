"""Load the local ByMimic MJLab command implementation as a project module."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _command_source_path() -> Path:
  """Return the command source, allowing an explicit override when needed."""
  default_path = (
    Path(__file__).resolve().parents[3]
    / "mimic"
    / "bymic_commands_mjlab_todo.py"
  )
  return Path(os.environ.get("BYMIC_COMMAND_SOURCE", default_path)).expanduser()


def _load_command_module():
  source_path = _command_source_path()
  if not source_path.is_file():
    raise FileNotFoundError(
      "ByMimic command source was not found at "
      f"{source_path}. Set BYMIC_COMMAND_SOURCE to bymic_commands_mjlab_todo.py."
    )

  module_name = "humanoid_hw6._bymic_commands_mjlab"
  module = sys.modules.get(module_name)
  if module is not None:
    return module

  spec = importlib.util.spec_from_file_location(module_name, source_path)
  if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load ByMimic command source: {source_path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = module
  spec.loader.exec_module(module)
  return module


_module = _load_command_module()
MotionCommand = _module.MotionCommand
MotionCommandCfg = _module.MotionCommandCfg

__all__ = ["MotionCommand", "MotionCommandCfg"]
