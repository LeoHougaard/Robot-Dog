# Robot Dog Simulation

This folder is for the URDF -> MuJoCo -> reinforcement-learning loop only.
Do not use it for real robot deployment or firmware changes.

## Simple Prototype Source

Actual Onshape package root:

```text
sim/robots/simple_quad_v0/onshape_export/assembly_1
```

Actual URDF:

```text
sim/robots/simple_quad_v0/onshape_export/assembly_1/urdf/assembly_1.urdf
```

Generated sim-specific MJCF:

```text
sim/robots/simple_quad_v0/mjcf/simple_quad_v0.xml
```

The URDF is the source of truth. The MJCF is generated under `sim/` and adds a
floating base, primitive collision geometry, conservative sim-only joint limits,
and ST3215-HS-style position actuators.

## Setup

Use Python 3.11-3.13 for the sim stack if possible; some MuJoCo/PyTorch wheels
may lag new Python releases.

```powershell
py -3.13 -m venv .venv-sim
.\.venv-sim\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r sim/requirements.txt
```

For Intel Arc XPU training, install a PyTorch build that exposes
`torch.xpu.is_available()` for your driver/runtime, then verify:

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install -r sim/requirements-xpu.txt
python sim/tools/check_torch_device.py
```

## Validation And Smoke Commands

Validate the nested Onshape package:

```powershell
python sim/tools/check_urdf_package.py sim/robots/simple_quad_v0/onshape_export/assembly_1
```

Generate the MuJoCo model:

```powershell
python sim/tools/build_simple_quad_mjcf.py
```

Load the MuJoCo model and optionally write a rendered PPM frame:

```powershell
python sim/tools/check_mujoco_model.py --steps 100
python sim/tools/check_mujoco_model.py --steps 100 --render sim/runs/simple_quad_check.ppm
```

Run random policy evaluation:

```powershell
python sim/evaluate.py --episodes 1 --max-steps 100
```

Start a short PPO run:

```powershell
python sim/train.py --total-timesteps 2048 --n-steps 128 --batch-size 64 --n-epochs 4
```

Collect PPO rollouts from multiple environments when the task is CPU-bound by
MuJoCo stepping. `--total-timesteps` is still the total transition count across
all envs, so this mainly reduces wall-clock time. Keep `--device cpu` for these
parallel rollout runs unless XPU has been validated for the specific command:

```powershell
python sim/train.py --task target --device cpu --num-envs 8 --n-steps 2048 --batch-size 512 --n-epochs 3 --verbose 0
```

The default vector backend for `--num-envs > 1` is subprocess-based. Use
`--vec-env dummy` if subprocess startup is a problem on a local machine.

Check whether a quiet or overnight run is still making progress:

```powershell
python sim/training_status.py --output-dir sim/runs/target_far_run_ppo
```

For unattended runs, pass `--checkpoint-freq 100000` or similar to `train.py`.
That writes restartable checkpoints while `monitor.csv` records episode
progress.

Run a supervised 8-hour far-target training session. The supervisor prevents
Windows sleep while it is alive, restarts from the newest checkpoint if a child
training process exits early, and keeps `target_far_overnight/latest/policy.zip`
updated for the viewer:

```powershell
python sim/overnight_target_train.py --hours 8 --output-dir sim/runs/target_far_overnight --num-envs 8 --n-steps 256 --batch-size 512 --n-epochs 3 --learning-rate 1e-5 --ent-coef 0.0 --checkpoint-freq 100000 --chunk-timesteps 10000000 --poll-seconds 30
```

Open the newest overnight policy after training:

```powershell
python sim/view_policy_tk.py --task target --policy ppo --checkpoint sim/runs/target_far_overnight/latest/policy.zip --terrain flat --episode-seconds 30 --target-radius-min 1.1 --target-radius-max 1.6 --success-radius 0.22 --target-velocity 0.30 --seed 1000
```

While the viewer is open, click `Refresh policy` or press `F5` to reload the
newest network saved by the overnight supervisor. Playback stays at the normal
50 Hz control rate, so the run is shown in regular time rather than training
speed.

Train a walking checkpoint from the built-in reference trot:

```powershell
python sim/train.py --task walk --device cpu --skip-ppo --pretrain-reference-steps 3000 --pretrain-epochs 40 --batch-size 256 --output-dir sim/runs/walk_pretrained
```

Evaluate the trained walker:

```powershell
$checkpoint = Get-ChildItem -Path sim/runs/walk_pretrained -Recurse -Filter policy.zip | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
python sim/evaluate.py --task walk --policy ppo --checkpoint "$checkpoint" --episodes 3 --max-steps 600
```

Render and open a contact sheet:

```powershell
python sim/render_policy.py --task walk --policy ppo --checkpoint "$checkpoint" --steps 600 --frames 6 --output sim/runs/walk_pretrained_contact_sheet.png
Start-Process sim/runs/walk_pretrained_contact_sheet.png
```

Train the target-reaching policy on random points. This starts with the
primitive target controller, then adds DAgger-style relabeling on the learned
policy's own rollout states to reduce drift. The current best run also trains
with sim-only servo target slew limiting plus torque, velocity, and joint
friction randomization:

```powershell
python sim/train.py --task target --device cpu --skip-ppo --pretrain-reference-steps 60000 --pretrain-epochs 100 --dagger-rounds 3 --dagger-steps 24000 --n-steps 512 --batch-size 512 --output-dir sim/runs/target_random_dagger_slew
```

The stronger current run uses:

```powershell
python sim/train.py --task target --device cpu --skip-ppo --pretrain-reference-steps 80000 --pretrain-epochs 120 --dagger-rounds 4 --dagger-steps 30000 --n-steps 512 --batch-size 512 --output-dir sim/runs/target_random_dagger_slew
```

Evaluate random target reaching:

```powershell
$targetCheckpoint = Get-ChildItem -Path sim/runs/target_random_dagger_slew -Recurse -Filter policy.zip | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
python sim/evaluate.py --task target --policy ppo --checkpoint "$targetCheckpoint" --episodes 20 --max-steps 700
```

On the current simple prototype, the improved scripted target controller reaches
`90/100` random targets on seed `1000`; the learned DAgger policy reaches
`89/100`. Remaining failures are hard backward/diagonal/side directions in the
rough generated model.

Drive by moving the target reward in the Tk MuJoCo renderer. Use this viewer on
Windows if the native OpenGL viewer opens as a blank white window:

```powershell
python sim/view_policy_tk.py --task target --policy ppo --checkpoint latest
```

Target-drive viewer controls:

- `W/S` or `Up/Down`: move the red target along world X.
- `A/D` or `Left/Right`: move the red target along world Y.
- `Space` or `P`: pause/resume.
- `R`: reset.
- `1`: trained target policy.
- `2`: primitive target controller.
- `3`: random actions.

To watch it continuously sample a new random target immediately after each
success:

```powershell
python sim/view_policy_tk.py --task target --policy ppo --checkpoint latest --auto-reset-success
```

Record and replay a random rollout:

```powershell
python sim/evaluate.py --episodes 1 --max-steps 200 --record sim/runs/random_rollout.npz
python sim/replay.py sim/runs/random_rollout.npz --render-rgb sim/runs/random_replay.ppm
```

## Longer First Training Command

After the smoke run is stable, try:

```powershell
python sim/train.py --total-timesteps 200000 --n-steps 512 --batch-size 256 --n-epochs 5 --output-dir sim/runs
```

For this prototype, prefer the DAgger target policy for interactive driving.
Short PPO fine-tunes have degraded this rough gait controller so far. Treat PPO
as experimental and evaluate it before using it in the viewer:

```powershell
python sim/train.py --task target --device cpu --pretrain-reference-steps 60000 --pretrain-epochs 100 --dagger-rounds 3 --dagger-steps 24000 --n-steps 512 --batch-size 256 --n-epochs 4 --learning-rate 5e-5 --ent-coef 0.0 --total-timesteps 50000 --output-dir sim/runs/target_ppo_experimental
```
