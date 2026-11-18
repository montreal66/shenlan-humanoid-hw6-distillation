from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ACCAD_YAML = (
  REPO_ROOT
  / "src/humanoid_hw6/config/g1/motion_data_cfg_g1_accad_walk.yaml"
)


def test_accad_motion_yaml_points_to_subset() -> None:
  text = ACCAD_YAML.read_text()
  assert "g1_accad_walk" in text


def test_accad_subset_has_25_clips() -> None:
  clip_dir = REPO_ROOT / "assets/motions/g1_accad_walk"
  npz_files = sorted(clip_dir.glob("*.npz"))
  assert len(npz_files) == 25


def test_humanoid_hw6_tasks_registered() -> None:
  from mjlab.tasks.registry import list_tasks

  import humanoid_hw6.config.g1  # noqa: F401

  tasks = set(list_tasks())
  expected = {
    "Mjlab-Humanoid-HW6-Teacher-G1",
    "Mjlab-Humanoid-HW6-Student-Action-Matching-G1",
    "Mjlab-Humanoid-HW6-Student-Direct-PPO-KL005-G1",
    "Mjlab-Humanoid-HW6-Student-Direct-PPO-Legacy-Hparams-KL01-G1",
    "Mjlab-Humanoid-HW6-Student-Direct-Action-Matching-G1",
    "Mjlab-Humanoid-HW6-Student-KL-Matching-G1",
  }
  assert expected.issubset(tasks)



def test_provided_teacher_checkpoint_exists() -> None:
  teacher_ckpt = REPO_ROOT / "checkpoints/g1_hw6_teacher/model_latest.pt"
  teacher_onnx = REPO_ROOT / "checkpoints/g1_hw6_teacher/g1_hw6_teacher.onnx"
  assert teacher_ckpt.is_file()
  assert teacher_onnx.is_file()


def test_student_runner_cfg_defaults_to_hw6_teacher() -> None:
  from humanoid_hw6.config.g1.rl_cfg import (
    DEFAULT_TEACHER_CHECKPOINT,
    unitree_g1_student_action_matching_runner_cfg,
    unitree_g1_student_kl_matching_runner_cfg,
  )

  assert DEFAULT_TEACHER_CHECKPOINT == "checkpoints/g1_hw6_teacher/model_latest.pt"
  assert (
    unitree_g1_student_action_matching_runner_cfg().teacher_checkpoint_file
    == DEFAULT_TEACHER_CHECKPOINT
  )
  assert (
    unitree_g1_student_kl_matching_runner_cfg().teacher_checkpoint_file
    == DEFAULT_TEACHER_CHECKPOINT
  )


def test_same_observation_distillation_requires_explicit_direct_teacher() -> None:
  from humanoid_hw6.config.g1.rl_cfg import (
    unitree_g1_student_direct_action_matching_runner_cfg,
    unitree_g1_student_action_matching_runner_cfg,
  )

  cfg = unitree_g1_student_direct_action_matching_runner_cfg()
  future_cfg = unitree_g1_student_action_matching_runner_cfg()
  assert cfg.teacher_checkpoint_file is None
  assert cfg.actor.class_name.endswith("SameObservationStudentTeacherActor")
  assert "teacher" not in cfg.obs_groups
  assert cfg.actor.distribution_cfg == future_cfg.actor.distribution_cfg
  assert cfg.algorithm == future_cfg.algorithm
  assert cfg.max_iterations == future_cfg.max_iterations


def test_direct_ppo_kl_ablation_changes_only_desired_kl() -> None:
  from humanoid_hw6.config.g1.rl_cfg import (
    unitree_g1_student_direct_ppo_kl005_runner_cfg,
    unitree_g1_student_direct_ppo_runner_cfg,
  )

  baseline_cfg = unitree_g1_student_direct_ppo_runner_cfg()
  ablation_cfg = unitree_g1_student_direct_ppo_kl005_runner_cfg()
  assert baseline_cfg.algorithm.desired_kl == 0.01
  assert ablation_cfg.algorithm.desired_kl == 0.005
  ablation_cfg.algorithm.desired_kl = baseline_cfg.algorithm.desired_kl
  assert ablation_cfg.algorithm == baseline_cfg.algorithm


def test_direct_ppo_legacy_hparams_kl01_cfg() -> None:
  from humanoid_hw6.config.g1.rl_cfg import (
    unitree_g1_student_direct_ppo_legacy_hparams_kl01_runner_cfg,
  )

  cfg = unitree_g1_student_direct_ppo_legacy_hparams_kl01_runner_cfg()
  assert cfg.actor.distribution_cfg["init_std"] == 0.4
  assert cfg.algorithm.learning_rate == 5.0e-4
  assert cfg.algorithm.entropy_coef == 0.0025
  assert cfg.algorithm.desired_kl == 0.01
  assert cfg.max_iterations == 20_000
