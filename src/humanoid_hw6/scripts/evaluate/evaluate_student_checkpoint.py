"""Evaluate a student checkpoint in a shared motion-tracking setup.

The evaluator always uses the current-reference-only Direct PPO environment.
It loads the common student portion of a checkpoint, so Action Matching, KL
Matching, and Direct PPO checkpoints can be compared without task-specific
arguments.  A distilled checkpoint's frozen ``teacher.*`` parameters are
intentionally excluded from evaluation.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from humanoid_hw6.mdp.motion.base import MotionCommand
from humanoid_hw6.rl.runner import _normalize_gaussian_distribution_state_dict

DIRECT_TASK_ID = "Mjlab-Humanoid-HW6-Student-Direct-PPO-G1"


@dataclass(frozen=True)
class Config:
  checkpoint_file: Path
  """Action-matching, KL-matching, or Direct PPO checkpoint to evaluate."""

  motion_file: Path = Path("src/humanoid_hw6/config/g1/motion_data_cfg_g1_accad_walk.yaml")
  """The shared motion-library YAML used by every evaluated policy."""

  num_envs: int = 128
  """Number of independently sampled episodes per repeat."""

  sampling_mode: Literal["start", "uniform"] = "uniform"
  """Fixed evaluation sampling; uniform matches the training phase distribution better."""

  action_mode: Literal["stochastic", "deterministic"] = "stochastic"
  """Use PPO's sampled Gaussian actions (training-style) or the deployed mean action."""

  environment_profile: Literal["training_like", "clean"] = "training_like"
  """Keep training randomization/noise, or remove it for a clean tracking diagnostic."""

  repeats: int = 3
  """Number of fixed-seed evaluation repeats to aggregate."""

  seed: int = 10_000
  """First evaluation seed; repeat i uses seed + i."""

  device: str | None = None
  """Defaults to cuda:0 when CUDA is available."""

  output_dir: Path | None = None
  """Defaults to <checkpoint directory>/evaluation/."""


def _set_seed(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def _load_student_state(checkpoint_file: Path) -> tuple[dict[str, torch.Tensor], str]:
  checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
  actor_state = checkpoint.get("actor_state_dict")
  if not isinstance(actor_state, dict):
    raise ValueError(
      "Checkpoint does not contain an `actor_state_dict`; expected an HW6 PPO checkpoint."
    )

  is_distilled = any(str(key).startswith("teacher.") for key in actor_state)
  student_state = {
    str(key): value
    for key, value in actor_state.items()
    if not str(key).startswith("teacher.")
  }
  _normalize_gaussian_distribution_state_dict(student_state)
  return student_state, "distilled" if is_distilled else "direct"


def _configure_evaluation(
  env_cfg,
  motion_file: Path,
  num_envs: int,
  seed: int,
  sampling_mode: Literal["start", "uniform"],
  environment_profile: Literal["training_like", "clean"],
) -> None:
  if not motion_file.is_file():
    raise FileNotFoundError(f"Motion library not found: {motion_file}")
  if num_envs <= 0:
    raise ValueError(f"`num_envs` must be positive, got {num_envs}.")

  env_cfg.seed = seed
  env_cfg.scene.num_envs = num_envs
  motion_command = env_cfg.commands["motion"]
  motion_command.motion_file = str(motion_file)
  motion_command.sampling_mode = sampling_mode

  if environment_profile != "clean":
    return

  motion_command.pose_range = {}
  motion_command.velocity_range = {}
  motion_command.joint_position_range = (0.0, 0.0)

  # Evaluation measures policy tracking, not robustness to random pushes,
  # observation noise, randomized COM, or randomized friction.
  env_cfg.events = {}
  for observation_group in env_cfg.observations.values():
    if observation_group is not None:
      observation_group.enable_corruption = False


def _clip_names(command: MotionCommand) -> list[str]:
  if command.motion_lib is None:
    return [Path(command.cfg.motion_file).stem] * command.num_envs
  return [
    command.motion_lib.motion_files[int(motion_id)].stem
    for motion_id in command.motion_ids.detach().cpu().tolist()
  ]


def _evaluate_repeat(
  cfg: Config,
  *,
  seed: int,
  device: str,
  student_state: dict[str, torch.Tensor],
) -> list[dict[str, object]]:
  _set_seed(seed)
  env_cfg = load_env_cfg(DIRECT_TASK_ID, play=False)
  agent_cfg = load_rl_cfg(DIRECT_TASK_ID)
  _configure_evaluation(
    env_cfg,
    cfg.motion_file.resolve(),
    cfg.num_envs,
    seed,
    cfg.sampling_mode,
    cfg.environment_profile,
  )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  try:
    runner_cls = load_runner_cls(DIRECT_TASK_ID)
    if runner_cls is None:
      raise RuntimeError(f"No runner registered for {DIRECT_TASK_ID}.")
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    missing, unexpected = runner.alg.actor.load_state_dict(student_state, strict=False)
    if missing or unexpected:
      raise ValueError(
        "Checkpoint student weights do not match the current HW6 StudentActor. "
        f"Missing keys: {missing}; unexpected keys: {unexpected}."
      )
    runner.alg.actor.eval()
    policy = runner.get_inference_policy(device=device)

    command = env.unwrapped.command_manager.get_term("motion")
    if not isinstance(command, MotionCommand):
      raise TypeError("Evaluation task did not create an HW6 MotionCommand.")
    clips = _clip_names(command)

    rewards = torch.zeros(cfg.num_envs, dtype=torch.float32, device=device)
    steps = torch.zeros(cfg.num_envs, dtype=torch.long, device=device)
    done = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)
    failed = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)
    obs = env.get_observations()

    while not bool(done.all()):
      active = ~done
      with torch.no_grad():
        if cfg.action_mode == "stochastic":
          # PPO and the two distillation algorithms collect rollouts with this
          # exact mode.  The seed set above makes the sampled comparison
          # reproducible across checkpoints.
          actions = runner.alg.actor(obs, stochastic_output=True)
        else:
          actions = policy(obs)
      obs, reward, dones, _ = env.step(actions)
      rewards[active] += reward[active]
      steps[active] += 1

      newly_done = dones.bool() & active
      if torch.any(newly_done):
        failed[newly_done] = env.unwrapped.termination_manager.terminated[newly_done]
        done |= newly_done

    rows: list[dict[str, object]] = []
    for index in range(cfg.num_envs):
      step_count = int(steps[index].item())
      episode_reward = float(rewards[index].item())
      is_failure = bool(failed[index].item())
      rows.append(
        {
          "seed": seed,
          "episode": index,
          "clip": clips[index],
          "return": episode_reward,
          "mean_step_reward": episode_reward / max(step_count, 1),
          "steps": step_count,
          "duration_s": step_count * float(env.unwrapped.step_dt),
          "failure": int(is_failure),
          "completion": int(not is_failure),
        }
      )
    return rows
  finally:
    env.close()


def _mean(rows: list[dict[str, object]], key: str) -> float:
  return sum(float(row[key]) for row in rows) / len(rows)


def run_evaluation(cfg: Config) -> dict[str, object]:
  configure_torch_backends()
  checkpoint_file = cfg.checkpoint_file.resolve()
  if not checkpoint_file.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
  if cfg.repeats <= 0:
    raise ValueError(f"`repeats` must be positive, got {cfg.repeats}.")

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  student_state, checkpoint_kind = _load_student_state(checkpoint_file)
  print(f"[INFO] Checkpoint type: {checkpoint_kind}")
  print(f"[INFO] Evaluation task: {DIRECT_TASK_ID}")
  print(f"[INFO] Device: {device}; repeats: {cfg.repeats}; episodes/repeat: {cfg.num_envs}")
  print(
    f"[INFO] Action mode: {cfg.action_mode}; "
    f"environment profile: {cfg.environment_profile}"
  )

  rows: list[dict[str, object]] = []
  for repeat in range(cfg.repeats):
    repeat_seed = cfg.seed + repeat
    print(f"[INFO] Running repeat {repeat + 1}/{cfg.repeats} (seed={repeat_seed})")
    rows.extend(
      _evaluate_repeat(
        cfg,
        seed=repeat_seed,
        device=device,
        student_state=student_state,
      )
    )

  summary: dict[str, object] = {
    "checkpoint": str(checkpoint_file),
    "checkpoint_type": checkpoint_kind,
    "evaluation_task": DIRECT_TASK_ID,
    "motion_file": str(cfg.motion_file.resolve()),
    "environment_profile": cfg.environment_profile,
    "action_mode": cfg.action_mode,
    "sampling_mode": cfg.sampling_mode,
    "repeats": cfg.repeats,
    "episodes_per_repeat": cfg.num_envs,
    "episodes_total": len(rows),
    "mean_return": _mean(rows, "return"),
    "mean_step_reward": _mean(rows, "mean_step_reward"),
    "mean_duration_s": _mean(rows, "duration_s"),
    "completion_rate": _mean(rows, "completion"),
    "failure_rate": _mean(rows, "failure"),
  }

  output_dir = cfg.output_dir or checkpoint_file.parent / "evaluation"
  output_dir.mkdir(parents=True, exist_ok=True)
  episodes_path = output_dir / "episodes.csv"
  summary_path = output_dir / "summary.json"
  with episodes_path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  summary_path.write_text(json.dumps(summary, indent=2) + "\n")

  print("\nEvaluation summary")
  for key in (
    "mean_return",
    "mean_step_reward",
    "mean_duration_s",
    "completion_rate",
    "failure_rate",
  ):
    print(f"  {key}: {float(summary[key]):.4f}")
  print(f"[INFO] Episode results: {episodes_path}")
  print(f"[INFO] Summary: {summary_path}")
  return summary


def main() -> None:
  cfg = tyro.cli(Config)
  run_evaluation(cfg)


if __name__ == "__main__":
  main()
