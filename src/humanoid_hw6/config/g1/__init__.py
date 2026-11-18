from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from humanoid_hw6.rl import OnPolicyRunner, StudentOnPolicyRunner

from .bymic_env_cfg import unitree_g1_bymic_env_cfg
from .env_cfgs import (
  unitree_g1_student_direct_ppo_env_cfg,
  unitree_g1_student_env_cfg,
  unitree_g1_teacher_env_cfg,
)
from .rl_cfg import (
  unitree_g1_student_action_matching_runner_cfg,
  unitree_g1_student_direct_action_matching_runner_cfg,
  unitree_g1_student_direct_ppo_runner_cfg,
  unitree_g1_student_direct_ppo_kl005_runner_cfg,
  unitree_g1_student_direct_ppo_legacy_hparams_kl01_runner_cfg,
  unitree_g1_student_kl_matching_runner_cfg,
  unitree_g1_teacher_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-ByMimic-Flat-Unitree-G1",
  env_cfg=unitree_g1_bymic_env_cfg(),
  play_env_cfg=unitree_g1_bymic_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Humanoid-HW6-Teacher-G1",
  env_cfg=unitree_g1_teacher_env_cfg(),
  play_env_cfg=unitree_g1_teacher_env_cfg(play=True),
  rl_cfg=unitree_g1_teacher_ppo_runner_cfg(),
  runner_cls=OnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Humanoid-HW6-Student-Action-Matching-G1",
  env_cfg=unitree_g1_student_env_cfg(),
  play_env_cfg=unitree_g1_student_env_cfg(play=True),
  rl_cfg=unitree_g1_student_action_matching_runner_cfg(),
  runner_cls=StudentOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Humanoid-HW6-Student-Direct-PPO-G1",
  env_cfg=unitree_g1_student_direct_ppo_env_cfg(),
  play_env_cfg=unitree_g1_student_direct_ppo_env_cfg(play=True),
  rl_cfg=unitree_g1_student_direct_ppo_runner_cfg(),
  runner_cls=OnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Humanoid-HW6-Student-Direct-PPO-KL005-G1",
  env_cfg=unitree_g1_student_direct_ppo_env_cfg(),
  play_env_cfg=unitree_g1_student_direct_ppo_env_cfg(play=True),
  rl_cfg=unitree_g1_student_direct_ppo_kl005_runner_cfg(),
  runner_cls=OnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Humanoid-HW6-Student-Direct-PPO-Legacy-Hparams-KL01-G1",
  env_cfg=unitree_g1_student_direct_ppo_env_cfg(),
  play_env_cfg=unitree_g1_student_direct_ppo_env_cfg(play=True),
  rl_cfg=unitree_g1_student_direct_ppo_legacy_hparams_kl01_runner_cfg(),
  runner_cls=OnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Humanoid-HW6-Student-Direct-Action-Matching-G1",
  env_cfg=unitree_g1_student_direct_ppo_env_cfg(),
  play_env_cfg=unitree_g1_student_direct_ppo_env_cfg(play=True),
  rl_cfg=unitree_g1_student_direct_action_matching_runner_cfg(),
  runner_cls=StudentOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Humanoid-HW6-Student-KL-Matching-G1",
  env_cfg=unitree_g1_student_env_cfg(),
  play_env_cfg=unitree_g1_student_env_cfg(play=True),
  rl_cfg=unitree_g1_student_kl_matching_runner_cfg(),
  runner_cls=StudentOnPolicyRunner,
)
