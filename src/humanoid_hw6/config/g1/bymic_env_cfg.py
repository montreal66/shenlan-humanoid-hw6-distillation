"""G1 tracking task that uses the local ByMimic motion command."""

from __future__ import annotations

import os
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

from humanoid_hw6.bymic_command import MotionCommandCfg

DEFAULT_BYMIC_MOTION_FILE = Path(__file__).resolve().parents[5] / "mimic" / "dance1_subject2.npz"


def unitree_g1_bymic_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the standard G1 tracking environment with ByMimic references."""
  cfg = unitree_g1_flat_tracking_env_cfg(play=play)
  source_motion = cfg.commands["motion"]
  motion_file = Path(os.environ.get("BYMIC_MOTION_FILE", DEFAULT_BYMIC_MOTION_FILE))
  if not motion_file.is_file():
    raise FileNotFoundError(
      f"ByMimic motion file not found: {motion_file}. "
      "Set BYMIC_MOTION_FILE or override --env.commands.motion.motion-file."
    )

  cfg.commands["motion"] = MotionCommandCfg(
    entity_name="robot",
    motion_file=str(motion_file),
    anchor_body_name=source_motion.anchor_body_name,
    body_names=source_motion.body_names,
    resampling_time_range=source_motion.resampling_time_range,
    pose_range=source_motion.pose_range,
    velocity_range=source_motion.velocity_range,
    joint_position_range=source_motion.joint_position_range,
    sampling_mode="start" if play else "adaptive",
  )
  return cfg
