"""RL configuration for the Unitree G1 Humanoid HW6 tasks."""

from mjlab.rl import RslRlModelCfg, RslRlPpoAlgorithmCfg

from humanoid_hw6.rl import (
  ActionMatchingPpoAlgorithmCfg,
  KlMatchingPpoAlgorithmCfg,
  OnPolicyRunnerCfg,
  StudentOnPolicyRunnerCfg,
)

DEFAULT_TEACHER_CHECKPOINT = "checkpoints/g1_hw6_teacher/model_latest.pt"


def _wandb_tags(*extra: str) -> tuple[str, ...]:
  return extra


def unitree_g1_teacher_ppo_runner_cfg() -> OnPolicyRunnerCfg:
  return OnPolicyRunnerCfg(
    seed=1,
    actor=RslRlModelCfg(
      class_name="humanoid_hw6.rl.models.teacher:TeacherActor",
      hidden_dims=(512, 512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "log",
      },
    ),
    critic=RslRlModelCfg(
      class_name="humanoid_hw6.rl.models.teacher:TeacherCritic",
      hidden_dims=(512, 512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_hw6_teacher",
    wandb_project="humanoid_hw6",
    wandb_tags=_wandb_tags("humanoid_hw6", "teacher", "privileged"),
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
    obs_groups={"actor": ("policy",), "critic": ("critic",)},
  )


def unitree_g1_student_direct_ppo_runner_cfg() -> OnPolicyRunnerCfg:
  """PPO baseline with the deployable student observation only.

  This uses the teacher PPO optimization settings while retaining only the
  current-reference student observation. It is therefore a fairer baseline
  for isolating the value of the teacher's future-reference input: it has no
  frozen teacher and no distillation loss.
  """
  return OnPolicyRunnerCfg(
    seed=1,
    actor=RslRlModelCfg(
      class_name="humanoid_hw6.rl.models.student:StudentActor",
      hidden_dims=(512, 512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "log",
      },
    ),
    critic=RslRlModelCfg(
      class_name="rsl_rl.models:MLPModel",
      hidden_dims=(512, 512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_hw6_student_direct_ppo",
    wandb_project="humanoid_hw6",
    wandb_tags=_wandb_tags("humanoid_hw6", "student", "direct_ppo"),
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
  )


def unitree_g1_student_direct_ppo_kl005_runner_cfg() -> OnPolicyRunnerCfg:
  """Direct-PPO ablation changing only the adaptive-PPO KL target."""
  cfg = unitree_g1_student_direct_ppo_runner_cfg()
  cfg.algorithm.desired_kl = 0.005
  cfg.experiment_name = "g1_hw6_student_direct_ppo_kl005"
  cfg.wandb_tags = _wandb_tags("humanoid_hw6", "student", "direct_ppo", "kl005")
  return cfg


def unitree_g1_student_direct_ppo_legacy_hparams_kl01_runner_cfg() -> OnPolicyRunnerCfg:
  """Old Direct-PPO optimization settings with the new KL target.

  The environment remains the current teacher-horizon-aligned Direct-PPO
  environment. This isolates the old optimizer/exploration settings from the
  historical run's clip-phase sampling difference.
  """
  cfg = unitree_g1_student_direct_ppo_runner_cfg()
  cfg.actor.distribution_cfg["init_std"] = 0.4
  cfg.algorithm.learning_rate = 5.0e-4
  cfg.algorithm.entropy_coef = 0.0025
  cfg.algorithm.desired_kl = 0.01
  cfg.max_iterations = 20_000
  cfg.experiment_name = "g1_hw6_student_direct_ppo_legacy_hparams_kl01"
  cfg.wandb_tags = _wandb_tags(
    "humanoid_hw6", "student", "direct_ppo", "legacy_hparams", "kl01"
  )
  return cfg


def unitree_g1_student_direct_action_matching_runner_cfg() -> StudentOnPolicyRunnerCfg:
  """Action-match a frozen current-reference Direct PPO teacher.

  This is a same-observation distillation control: both policies receive the
  deployable current-reference actor observation. Its optimization and MSE
  schedule match the future-teacher action-matching experiment; the teacher
  observation and checkpoint source are the only intentional differences.
  The checkpoint is intentionally left unset so callers must explicitly select
  the converged Direct PPO checkpoint they wish to clone.
  """
  return StudentOnPolicyRunnerCfg(
    seed=1,
    actor=RslRlModelCfg(
      class_name=(
        "humanoid_hw6.rl.models.student_teacher:SameObservationStudentTeacherActor"
      ),
      hidden_dims=(512, 512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.4,
        "std_type": "log",
      },
    ),
    critic=RslRlModelCfg(
      class_name="rsl_rl.models:MLPModel",
      hidden_dims=(512, 512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=ActionMatchingPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.0025,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=5.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.005,
      max_grad_norm=1.0,
      bc_coef_start=1.0,
      bc_coef_end=0.05,
      bc_anneal_iters=20_000,
      bc_loss_type="mse",
    ),
    experiment_name="g1_hw6_student_direct_action_matching",
    wandb_project="humanoid_hw6",
    wandb_tags=_wandb_tags("humanoid_hw6", "student", "direct_action_matching"),
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=20_000,
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    teacher_checkpoint_file=None,
  )


def unitree_g1_student_action_matching_runner_cfg() -> StudentOnPolicyRunnerCfg:
  return StudentOnPolicyRunnerCfg(
    seed=1,
    actor=RslRlModelCfg(
      class_name="humanoid_hw6.rl.models.student_teacher:StudentTeacherActor",
      hidden_dims=(512, 512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.4,
        "std_type": "log",
      },
    ),
    critic=RslRlModelCfg(
      class_name="rsl_rl.models:MLPModel",
      hidden_dims=(512, 512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=ActionMatchingPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.0025,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=5.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.005,
      max_grad_norm=1.0,
      bc_coef_start=1.0,
      bc_coef_end=0.05,
      bc_anneal_iters=20_000,
      bc_loss_type="mse",
    ),
    experiment_name="g1_hw6_student_action_matching",
    wandb_project="humanoid_hw6",
    wandb_tags=_wandb_tags("humanoid_hw6", "student", "action_matching"),
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=20_000,
    obs_groups={
      "actor": ("actor",),
      "critic": ("critic",),
      "teacher": ("teacher_policy",),
    },
    teacher_checkpoint_file=DEFAULT_TEACHER_CHECKPOINT,
  )


def unitree_g1_student_kl_matching_runner_cfg() -> StudentOnPolicyRunnerCfg:
  return StudentOnPolicyRunnerCfg(
    seed=1,
    actor=RslRlModelCfg(
      class_name="humanoid_hw6.rl.models.student_teacher:StudentTeacherActor",
      hidden_dims=(512, 512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.4,
        "std_type": "log",
      },
    ),
    critic=RslRlModelCfg(
      class_name="rsl_rl.models:MLPModel",
      hidden_dims=(512, 512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=KlMatchingPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=3.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.008,
      max_grad_norm=1.0,
      kl_coef=0.1,
      kl_coef_min=0.01,
      kl_coef_anneal_iters=60_000,
    ),
    experiment_name="g1_hw6_student_kl_matching",
    wandb_project="humanoid_hw6",
    wandb_tags=_wandb_tags("humanoid_hw6", "student", "kl_matching"),
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=20_000,
    obs_groups={
      "actor": ("actor",),
      "critic": ("critic",),
      "teacher": ("teacher_policy",),
    },
    teacher_checkpoint_file=DEFAULT_TEACHER_CHECKPOINT,
  )
