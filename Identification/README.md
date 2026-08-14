# LQR Identification Experiment

The micro-coaxial physical-domain Sim2Real design, scale derivations, conditional
trim-feasibility sampling, LQI weights, identification results, and paired
nonlinear control audit are documented in
[SIM2REAL_MICRO_DESIGN_zh.md](SIM2REAL_MICRO_DESIGN_zh.md). The current learned
gain scheduler is `shadow_only`: it produces finite candidate gains, but the
paired nonlinear audit has not shown a convergence advantage over the fixed
nominal LQI.

The follow-up [composite servo experiment](SIM2REAL_COMPOSITE_SERVO_EXPERIMENT_zh.md)
does not separately identify servo effectiveness and time constant. It predicts
the command-to-angular-acceleration step response at five short-time snapshots,
projects it onto fixed stable filters, and synthesizes a 19-state LQI. On the
held-out paired nonlinear audit, a 50% gain blend improves convergence by 1.34
percentage points over the matching fixed composite nominal controller. This is
still `research_only`; runtime gain updates remain disabled until time-varying
gain interpolation, uncertainty gating, sensor-path testing, and HIL validation
are complete.

The next-stage [offline flight-log design](OFFLINE_LOG_IDENTIFICATION_zh.md)
targets post-flight inference on complete real logs rather than onboard or
online adaptation. Its training data remains closed loop: stratified multiaxis
initial attitude/rate disturbances are allowed, while per-actuator steps,
open-loop sweeps, and oracle-gain data collection are explicitly excluded.
The canonical NPZ adapter and offline analysis CLI produce candidate composite
gains but never mark them as flight-accepted.

The angular-velocity neural-controller line (oracle per-airframe LQI distilled
into a causal student) is documented in
[DAgger_ANGULAR_VELOCITY_CONTROLLER_REPORT_zh.md](DAgger_ANGULAR_VELOCITY_CONTROLLER_REPORT_zh.md).
Pure behavior cloning failed in closed loop (safety 34-42%); a DAgger loop that
relabels student-visited states with the oracle LQI lifts unseen-airframe
closed-loop safety to 94.6% (oracle LQI 93.9%, fixed nominal LQI 96.0%) with
0.41 ms/step inference. GRU-192x2 is the recommended deployable candidate;
GRU/LSTM/Transformer close-loop at the same safety level, while a fixed-window
TCN lags. Runtime entry points: `flight-identification-lqi-gru-dagger`,
`flight-identification-lqi-gru-closed-loop`,
`flight-identification-lqi-gru-arch-compare`.

The offline analysis line is complete: MLP (100 Hz), TCN, and bidirectional GRU
are all retrained on the stratified `lqr_sim2real_micro_offline_logs_v2` data
with per-flight-count 1/2/4/8 comparisons. The selected hybrid is
TCN(roll/pitch) + MLP(yaw) (`sim2real_offline_logs_step_response_tcn_500hz_v6`
and `sim2real_offline_logs_step_response_mlp_v6`). Strict paired nonlinear
audits on all 860 unseen parameter groups x 8 initial conditions, repeated over
two evaluation seeds, give +9.5 to +10.1 percentage points convergence over the
fixed composite nominal LQI with a group-clustered 95% CI excluding zero,
non-inferior safety, and reduced saturation. Win/loss is positive in every
log-information and input-OOD quartile. Artifacts remain
`deployment_mode=research_only` with `gain_updates_enabled=false`; the 90%
analysis blend plus validation-calibrated log-quality thresholds are the
recommended HIL candidate inputs.

The deployment-contract line now uses the four-output LQR structure: the upper
rotor is owned by a virtual-pilot altitude hold and the identified LQR gain has
four outputs (lower motor + three servos). It was trained on
`lqr_sim2real_micro_offline_logs_v4` with MLP v7 (yaw) + TCN v7 (roll) + BiGRU
v7b (pitch). On 949 unseen groups x 8 initial conditions x 2 s over two seeds,
the 80% gain blend reaches 79.1%/78.7% convergence versus 68.5%/68.9% for the
fixed composite nominal (+10.6/+9.8 pp, clustered 95% CI excluding zero), safety
non-inferior, and 100% convergence on the 8 s long-duration audit. See
[OFFLINE_4OUT_COMPOSITE_REPORT_zh.md](OFFLINE_4OUT_COMPOSITE_REPORT_zh.md).

The consolidated conclusions, with deployment-feasibility analysis and the full
performance comparison, are in
[OFFLINE_COMPOSITE_IDENTIFICATION_REPORT_zh.md](OFFLINE_COMPOSITE_IDENTIFICATION_REPORT_zh.md).

The reproducible baseline uses 4,096 airframes with eight trials each:

```bash
env PYTHONNOUSERSITE=1 \
  PYTHONPATH=Identification/src:Controller/src:../TeleaiTrans_Simenv/src \
  /home/ryan/miniconda3/envs/rl-flight/bin/python \
  -m flight_identification.repeated_trial_experiment \
  --config Identification/configs/lqr_sim2real_micro_repeated8_v1.yaml

env PYTHONNOUSERSITE=1 \
  PYTHONPATH=Identification/src:Controller/src:../TeleaiTrans_Simenv/src \
  /home/ryan/miniconda3/envs/rl-flight/bin/python \
  -m flight_identification.repeated_trial_training \
  --dataset Identification/datasets/lqr_sim2real_micro_repeated8_v1 \
  --output Identification/runs/sim2real_micro_repeated8_mlp_v3 \
  --device cuda:0 --downsample 5

env PYTHONNOUSERSITE=1 \
  PYTHONPATH=Identification/src:Controller/src:../TeleaiTrans_Simenv/src \
  /home/ryan/miniconda3/envs/rl-flight/bin/python \
  -m flight_identification.sim2real_control_evaluation \
  --checkpoint Identification/runs/sim2real_micro_repeated8_mlp_v3/identifier.pt \
  --dataset Identification/datasets/lqr_sim2real_micro_repeated8_v1 \
  --experiment-config Identification/configs/lqr_sim2real_micro_repeated8_v1.yaml \
  --output Identification/runs/sim2real_micro_repeated8_mlp_v3/control_audit_paired_large_v2.json \
  --device cuda:0 --maximum-parameter-groups 393 \
  --evaluation-initial-conditions 8
```

The evaluator rebuilds SimEnv for every controller variant with identical seeds,
sample ordering, initial states, and noise counters. Its confidence intervals
cluster repeated initial conditions by airframe parameter group. The companion
`sim2real_parameter_diagnostics` command reports which inferred quantities are
actually consumed by gain synthesis and separates identifiability from local
closed-loop sensitivity.

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
PYTHONPATH=Identification/src:Controller/src:../TeleaiTrans_Simenv/src \
python -m flight_identification \
  --config Identification/configs/lqr_zero_attitude_yaw_rate_v1.yaml
```

A small pipeline check can be generated without editing the YAML:

```bash
PYTHONPATH=Identification/src:Controller/src:../TeleaiTrans_Simenv/src \
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
  PYTHONPATH=Identification/src:Controller/src:../TeleaiTrans_Simenv/src \
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

The wide `[-2, 2]` randomization is retained only as a historical stress test.
The deployment-oriented repeated-trial design now uses explicit empirical-core
ranges for inertia, thrust-to-weight ratio, reaction authority, grid authority,
and actuator time constants. It deliberately excludes undefined coupling
scalars. See [EMPIRICAL_PRIOR_DESIGN_zh.md](EMPIRICAL_PRIOR_DESIGN_zh.md) for
the parameter audit, pilot fit, and gain/observer ablation.

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
PYTHONPATH=Identification/src:Controller/src:../TeleaiTrans_Simenv/src \
python -m flight_identification.control_evaluation \
  --checkpoint Identification/artifacts/mlp_baseline/identifier.pt \
  --dataset Identification/datasets/lqr_zero_attitude_yaw_rate_v1 \
  --output Identification/artifacts/mlp_baseline/control_value.json
```

The report compares nominal, identified, and oracle gains using true-plant pole
radii and infinite-horizon LQR cost. This is a local linear audit, not a
substitute for nonlinear gain-blending and saturation tests.

## Wide-Range Deployment Experiment

`lqr_wide_adaptation_v2.yaml` covers every raw randomized parameter over
`log10(scale) in [-2, 2]`. It contains 32,768 independent parameter groups and
65,536 one-second roll/pitch-zero, yaw-rate-zero episodes. A window enters the
adaptation dataset when it remains inside the safety envelope for the complete
first second; final convergence is metadata rather than an inclusion filter.

The selected identifier is a five-member `512-256-128` MLP ensemble trained on
14 LQR-effective log ratios. Each member has 883,342 parameters. The ensemble
artifact contains per-window calibrated uncertainty and is intentionally
separate from the final nonlinear-validation decision.

The online API consumes exactly one raw 500 Hz history:

```python
from flight_identification.deployment import AdaptiveLQRScheduler

scheduler = AdaptiveLQRScheduler(
    "Identification/artifacts/lqr_wide_effective_deployment_v2/"
    "deployment_validated.pt",
    device="cuda:0",
)
result = scheduler.schedule(history)  # history: [batch, 500, 14]
# Feed result.target_gain to the configured two-second interpolation loop.
```

Always use `result.accepted`; do not apply `predicted_gain` directly. A failed
or ill-conditioned DARE is rejected per vehicle and falls back to the nominal
gain without aborting the batch.

The current validated artifact is `shadow_only`, so `result.accepted` is always
false. This is deliberate: the local-linear non-degradation gate passed, but
paired 4 s and 8 s nonlinear tests did not improve convergence. Shadow mode is
appropriate for collecting flight histories and comparing predictions; it is
not authorization to update flight gains.

The frozen operating policy is described by
`configs/lqr_identifier_deployment_v2.yaml`. It requires a two-second gain
interpolation, at most one inference per second, and freezes adaptation on
invalid state, excessive tilt/rate, non-finite history, or excessive command
saturation.

## Repeated Destructive-Trial Calibration

The repeated-trial experiment uses a different deployment contract from the
online one-second scheduler above. One unknown airframe may perform many
independent identification flights, including flights that cross the safety
envelope. Every valid pre-failure prefix is retained. Identification and DARE
synthesis run once after the trials, and the resulting fixed gain is installed
before the first control step of a new evaluation flight.

Generate and train the 16-trial dataset with:

```bash
env PYTHONNOUSERSITE=1 \
  PYTHONPATH=Identification/src:Controller/src:../TeleaiTrans_Simenv/src \
  /home/ryan/miniconda3/envs/rl-flight/bin/python \
  -m flight_identification.repeated_trial_experiment \
  --config Identification/configs/lqr_repeated_trials_wide16_v1.yaml

env PYTHONNOUSERSITE=1 \
  PYTHONPATH=Identification/src \
  /home/ryan/miniconda3/envs/rl-flight/bin/python \
  -m flight_identification.repeated_trial_training \
  --dataset Identification/datasets/lqr_repeated_trials_wide16_v1 \
  --output Identification/artifacts/lqr_repeated_trials_wide16_mlp_v1_seed20260814 \
  --device cuda:0
```

The one-shot runtime API consumes raw repeated histories and their pre-failure
masks:

```python
from flight_identification.repeated_trial_deployment import (
    RepeatedTrialLQRScheduler,
)

scheduler = RepeatedTrialLQRScheduler("path/to/identifier.pt", device="cuda:0")
result = scheduler.synthesize(histories, valid_mask)
# histories: [batch, trials, 1000, 14] at 500 Hz
# result.predicted_gain: [batch, 5, 10]
# result.servo_time_constant_s updates the command-driven servo observer.
```

This artifact has no automatic acceptance gate and
`scheduler.deployment_validated` is false. The gain must not be applied without
the separate nonlinear per-airframe validation described in
[`REPEATED_TRIAL_RESULTS_zh.md`](REPEATED_TRIAL_RESULTS_zh.md). The evaluation
showed that gain synthesis alone is insufficient: without servo-angle feedback,
the command observer must also model known deadzone, backlash and rate limits.
