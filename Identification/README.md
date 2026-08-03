# LQR Identification Experiment

This package generates supervised one-second histories for an online LQR model
identifier. The flight controller always starts from one nominal gain. Only the
SimEnv plant receives randomized parameters, so the collected distribution does
not leak an oracle gain computed from the parameter label.

The experiment tracks zero roll/pitch and zero yaw rate. It does not track yaw
angle. The LQR receives a first-order servo state propagated only from previous
servo commands and nominal actuator dynamics. True `servo_angle` is neither a
controller input nor a dataset feature.

The default config uses ideal rate and motor-speed measurements to isolate
structural learnability. The deployment-oriented
`lqr_zero_attitude_yaw_rate_sensor_v1.yaml` config routes SimEnv gyro and motor
sensor noise/delay into both the controller and dataset. Attitude is still a
truth-tilt proxy because SimEnv does not yet implement an attitude estimator;
the manifest states this limitation explicitly.

## Run

From the repository root:

```bash
PYTHONPATH=Identification/src:Controller/src:SimEnv/src \
python -m flight_identification \
  --config Identification/configs/lqr_zero_attitude_yaw_rate_v1.yaml
```

A small pipeline check can be generated without editing the YAML:

```bash
PYTHONPATH=Identification/src:Controller/src:SimEnv/src \
python -m flight_identification \
  --config Identification/configs/lqr_zero_attitude_yaw_rate_v1.yaml \
  --parameter-groups 16 \
  --parallel-count 16 \
  --output-directory /tmp/lqr-identification-smoke
```

For range and sample-count ablations, the generator also accepts
`--initial-conditions-per-group N` and
`--log10-effectiveness-range LOW HIGH`. When the initial-condition count is
changed, `--parallel-count` is rounded down to a whole number of parameter
groups.

The output directory must be empty. Successful episodes are windowed into
`train/`, `validation/`, and `test/`. Failed episodes are excluded from those
regression windows but retained in `failures/` with a reason code.

## GPU Execution

This host has a V100 (`sm_70`) and the project CUDA environment is
`/home/ryan/miniconda3/envs/rl-flight`. Do not use `/usr/bin/python3`: it can
load the user-site PyTorch `2.13.0+cu130`, which does not support this V100 and
driver combination. Disable the user site explicitly and invoke the environment
by absolute path:

```bash
env PYTHONNOUSERSITE=1 \
  PYTHONPATH=Identification/src:Controller/src:SimEnv/src \
  /home/ryan/miniconda3/envs/rl-flight/bin/python \
  -m flight_identification \
  --config Identification/configs/lqr_zero_attitude_yaw_rate_v1.yaml \
  --device cuda:0
```

Training uses the same interpreter and accepts `--device cuda:0`:

```bash
env PYTHONNOUSERSITE=1 \
  PYTHONPATH=Identification/src \
  /home/ryan/miniconda3/envs/rl-flight/bin/python \
  -m flight_identification.training \
  --dataset Identification/datasets/lqr_zero_attitude_yaw_rate_v1 \
  --output Identification/artifacts/mlp_baseline \
  --target-mode lqr_effective \
  --device cuda:0
```

When Codex runs these commands, GPU device passthrough requires host execution
rather than the default filesystem sandbox. The verified host stack is PyTorch
`2.12.1+cu126`, CUDA wheel `12.6`, V100 capability `(7,0)`, with `sm_70`
available. A real CUDA forward/backward pass and an end-to-end identifier
training smoke test both passed on 2026-08-03.

## Dataset Contract

Each successful shard is a `torch.save` mapping containing:

- `features`: `[N,500,14]` one-second histories at 500 Hz;
- `labels`: `[N,13]` base-10 logarithms of multiplicative nominal scales;
- `group_id`: the physical-parameter identity used for leakage-free splitting;
- `episode_id`: the initial-condition rollout identity;
- `window_start_step`: the offset within the source episode;
- `controller_servo_observer_initial`: command-observer state at the window edge;
- `information_score`: angular-rate RMS plus actuator-command movement.

Parameter groups are assigned to a split before initial conditions are created
and before episodes are windowed. All windows and initial conditions sharing one
parameter label therefore remain in exactly one split.

The randomized labels are three principal-inertia scales, a collective
mass/thrust scale that preserves hover speed, a common motor reaction-torque
scale, two motor time-constant scales, three grid-effectiveness scales, and
three servo time-constant scales. Principal inertia samples are rejected until
the rigid-body triangle inequalities hold.

Failure records should be used later for feasibility/OOD supervision. They are
not silently deleted even though they do not enter the normal parameter
regression dataset.

## MLP Feasibility Baseline

The baseline downsamples each one-second history to 100 Hz, normalizes each
feature channel using the training split only, and trains a flattened-history
MLP. Sampling is balanced by parameter group, and reported test metrics include
group-averaged predictions so repeated initial conditions do not inflate the
result.

```bash
PYTHONPATH=Identification/src \
python -m flight_identification.training \
  --dataset Identification/datasets/lqr_zero_attitude_yaw_rate_v1 \
  --output Identification/artifacts/mlp_baseline \
  --downsample 5 \
  --maximum-start-s 1.0
```

`report.json` includes per-parameter log-space RMSE, `R²`, and multiplicative
median/p90 errors for window-level and parameter-group-level predictions. The
train-mean baseline is recorded alongside the learned model.

Use `--target-mode lqr_effective` to predict the quantities that determine the
local LQR model: nonzero entries of angular acceleration per actuator state and
the five actuator time constants. This removes two exact ambiguity directions
in the raw physical labels: common moment/inertia scaling, and the trade between
collective thrust scale and all three grid-gain scales.

An effective-target checkpoint can be audited for actual control value by
reconstructing a gain from its predictions and applying that gain to every true
test-group linearization:

```bash
PYTHONPATH=Identification/src:Controller/src:SimEnv/src \
python -m flight_identification.control_evaluation \
  --checkpoint Identification/artifacts/mlp_baseline/identifier.pt \
  --dataset Identification/datasets/lqr_zero_attitude_yaw_rate_v1 \
  --output Identification/artifacts/mlp_baseline/control_value.json
```

The report compares nominal, identified, and oracle gains using true-plant pole
radii and infinite-horizon LQR cost. This is a local linear audit, not a
substitute for nonlinear gain-blending and saturation tests.
