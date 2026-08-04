# Bad-point angular-acceleration controller results

## Scope

This experiment selected three parameter combinations from the
`sim2real_micro` identification range where a controller designed for the
nominal plant converged poorly but an oracle LQI designed for the actual plant
still converged. Each selected plant was materialized as an exact SimEnv
configuration with a nonlinear five-actuator hover allocation.

The neural controller is point-specific. Its inner loop maps angular-
acceleration tracking state to three residual actions around the allocated
hover command. The outer loop is a PID attitude/rate cascade. The comparison
controller is the nominal-model LQI and controls all five physical actuators.

## Selected points

The screening used 64 paired initial conditions per point for 2 seconds. The
fractions below are the number of trials that met the convergence criteria.

| Group | Nominal LQI | Oracle LQI | Oracle gain | Upper/lower hover | Servo hover trim |
| --- | ---: | ---: | ---: | --- | --- |
| 1927 | 0.0313 | 0.8750 | 0.8438 | 0.4635 / 0.4365 | -0.3938, 0.0664, -0.2605 |
| 1581 | 0.0469 | 0.9844 | 0.9375 | 0.2708 / 0.5866 | 0.7799, 0.5692, 0.0000 |
| 3468 | 0.1406 | 0.8125 | 0.6719 | 0.7290 / 0.6707 | 0.0430, -0.2758, 0.3500 |

All three allocated hover states passed the static nonlinear validation.

## Inner-loop training

The retained checkpoints were initialized from the v8 feature encoder and
supervised with a local incremental nonlinear-dynamic-inversion teacher. The
direct fixed angular-acceleration suite produced these scores:

| Group | Examples | Final action RMSE | Direct inner-loop score |
| --- | ---: | ---: | ---: |
| 1927 | 1,536,000 | 0.00223 | 46.6143 |
| 1581 | 1,536,000 | 0.01120 | 40.9950 |
| 3468 | 1,536,000 | 0.00829 | 40.2272 |

For group 1927, subsequent SAC fine-tuning degraded the fixed score from
46.6143 to 42.979 at 3.15M steps and 40.296 at 4.19M steps. Those SAC
checkpoints are rejected; the supervised checkpoint is retained. Group 1581
needed an additional 2,560,000-example closed-loop DAgger pass ending at 100%
neural execution and action RMSE 0.00640.

## Full cascade comparison

The common score below renormalizes survival, tracking, and response from the
same eight-scenario `fixed_small_command_tracking_v1` suite. It excludes the
action term because the neural controller reports a three-dimensional residual
while LQI reports a normalized five-dimensional physical command. Raw total
scores remain available in the machine-readable report.

| Group | Neural common | Nominal LQI common | Delta | Winner | Minimum survival |
| --- | ---: | ---: | ---: | --- | ---: |
| 1927 | 82.1638 | 67.4483 | +14.7154 | Neural | 1.0000 |
| 1581 | 51.1191 | 68.3561 | -17.2370 | Nominal LQI | 0.9487 |
| 3468 | 79.2411 | 77.3027 | +1.9383 | Neural | 1.0000 |

The neural controller wins two of three points, but the mean common-score delta
is -0.1944. Therefore the experiment does not establish an aggregate advantage
over nominal LQI across the selected range.

The 1927 result is a strong point-specific advantage: the neural hover
roll/pitch RMSE is 0.102 degrees versus 4.689 degrees for nominal LQI. The 3468
advantage is modest and is concentrated in hover and roll/pitch step cases;
nominal LQI remains better on the circle and yaw-rate cases.

Group 1581 is a retained negative result. Its hover allocation requires a lower
to upper motor ratio of 2.166 and two large servo trims. The three-residual
neural action contract saturates in the cascade and cannot reproduce the full
five-actuator authority used by LQI. DAgger improves the original cascade raw
score from 46.39 to 50.53, but does not make the controller qualified: it loses
17.24 common-score points and terminates early in the combined step scenario.

## Decision

- Retain `group_1927_v1.pt` with the fast outer loop as the demonstrated neural
  winner.
- Retain `group_3468_v1.pt` with the fast outer loop as a marginal winner that
  still needs stronger yaw/circle validation before deployment.
- Do not deploy `group_1581_cascade_v1.pt`. Treat the point as evidence that the
  current three-residual action contract is insufficient near extreme hover
  allocations; a five-degree-of-freedom policy/allocation redesign is required
  before repeating this point.

Machine-readable aggregate results are in
`evaluation_runs/sim2real_bad_point_comparison_v1/report.json`. Screening data
is in
`../Identification/runs/sim2real_micro_bad_point_validation_pool_v1.json`, and
the exact selected configurations and hover allocations are in
`../Identification/runs/sim2real_micro_selected_points_v1.json`.
