# Humanoid HW6 Scripts

Utility scripts kept for the teaching workflow.

## Preview reference motions (Viser)

Browse the 25 ACCAD walk NPZ clips in a web viewer (no policy, reference pose only):

```bash
uv run python -m humanoid_hw6.scripts.data.visualize_motion_curate_viser \
  --motion assets/motions/g1_accad_walk
```

See [`ASSIGNMENT.md`](../../../ASSIGNMENT.md) §3.7 for full usage.

## Export checkpoint to ONNX

```bash
uv run python -m humanoid_hw6.scripts.deploy.export_checkpoint_to_onnx --help
```

## Play checkpoint helper

```bash
uv run play Mjlab-Humanoid-HW6-Teacher-G1 --help
```

## Evaluate a student checkpoint

Evaluate Action Matching, KL Matching, or Direct PPO in the same
current-reference-only environment. The script automatically loads the shared
student weights from distilled checkpoints and excludes their frozen teacher.
By default it uses fixed-seed uniform clip/phase sampling, the same observation
noise/domain randomization as training, and PPO's stochastic Gaussian actions.
This is the appropriate mode for matching the training metrics. Use
`--action-mode deterministic --environment-profile clean` only as a separate
deployment/clean-tracking diagnostic. Pass `--sampling-mode start` to evaluate
only clip beginnings.

```bash
uv run python -m humanoid_hw6.scripts.evaluate.evaluate_student_checkpoint \
  --checkpoint-file /path/to/model_latest.pt \
  --repeats 3 \
  --num-envs 128
```

## Instructor solution verification

```bash
uv run python scripts/verify_instructor_solutions.py
```
