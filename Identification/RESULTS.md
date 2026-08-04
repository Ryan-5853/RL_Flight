# Supervised LQR Identification Feasibility Results

Date: 2026-08-03

## Question

Can a one-second closed-loop flight history identify parameters well enough to
schedule an LQR controller, when yaw angle and servo angle are unavailable?

## Experimental Controls

- The plant parameters are randomized, while every rollout starts with one
  fixed nominal LQR gain.
- Roll/pitch track zero and yaw angular rate tracks zero.
- Features contain yaw-free tilt quaternion, body rate, measured motor speed,
  and the five applied commands. True servo angle and yaw angle are absent.
- Splits are disjoint by parameter group before initial-condition rollout and
  windowing. Metrics are reported after averaging predictions per parameter
  group as well as per window.
- The main medium-range experiment randomizes all labels jointly over
  `log10(scale) in [-0.5, 0.5]`, or approximately `[0.316, 3.16]` times nominal.
- Only converged episodes enter regression, as requested. Failed episodes remain
  in separate shards.

## Datasets

| Dataset | Range | Groups | Episodes | Converged | Windows |
| --- | ---: | ---: | ---: | ---: | ---: |
| broad ideal | `[0.1, 10]` | 128 | 1,024 | 438 (42.8%) | 1,752 |
| medium ideal | `[0.316, 3.16]` | 512 | 4,096 | 2,965 (72.4%) | 11,860 |
| medium sensors | `[0.316, 3.16]` | 256 | 2,048 | 1,701 (83.1%) | 6,804 |

The convergence rates across the two medium datasets are not directly
comparable because changing the group count changes the sampled label set.
`medium sensors` routes SimEnv gyro noise/delay and motor-speed noise into both
the controller and features. Attitude remains an explicitly marked truth-tilt
proxy because SimEnv has no attitude estimator.

## Structural Identifiability

The raw physical labels are over-parameterized for the local LQR model. Around
hover, the observable servo effectiveness contains

```text
collective_thrust_scale * grid_gain_scale / axis_inertia_scale
```

and motor yaw effectiveness contains

```text
reaction_torque_scale / yaw_inertia_scale
```

Consequently, common moment/inertia scaling and collective/grid tradeoff are two
ambiguity directions. Raw physical parameters can receive statistically useful
point estimates under a fixed training prior, but they are not uniquely
recoverable from these histories. The effective target replaces them with the
nine nonzero entries of angular acceleration per actuator state plus five
actuator time constants, which are the quantities used by the local LQR model.

## Main Fit Results

All main models use the first two one-second windows, 100 Hz histories, and
parameter-group-balanced sampling.

| Target/model | Parameters | Window mean R2 | Group mean R2 | Group log10 RMSE |
| --- | ---: | ---: | ---: | ---: |
| raw physical, `512-256-128` | 883k | 0.107 | 0.317 | 0.228 |
| LQR effective, `64-32` | 92k | 0.283 | 0.474 | 0.262 |
| LQR effective, `256-128-64` | 401k | 0.336 | 0.541 | 0.237 |
| LQR effective, `512-256-128` | 883k | 0.337 | 0.546 | 0.238 |

The 401k and 883k models are effectively tied. Capacity above roughly 400k is
not the current bottleneck.

For the 401k model, test-group mean R2 scales with independent training groups:

| Training groups | Training windows | Group mean R2 |
| ---: | ---: | ---: |
| 50 | 610 | 0.357 |
| 100 | 1,226 | 0.403 |
| 200 | 2,454 | 0.466 |
| 346 | 4,216 | 0.541 |

This learning curve has not saturated, so more independent plants should help.
Repeated windows from the same plant are not a substitute for new parameter
groups.

## Time And Sampling Ablations

- With a model trained on the first two windows, group mean R2 is `0.566` on
  `0-1 s` and `0.424` on `1-2 s`. The initial recovery transient carries most
  information.
- Training only `0-1 s` gives mean R2 `0.562`. Motor tau R2 improves to
  `0.54/0.63`, while the three servo tau values remain `0.04/0.07/-0.12`.
- Keeping all 500 Hz points increases the MLP to 1.83M parameters but gives mean
  R2 `0.555`; servo tau remains `0.09/-0.06/-0.09`. The servo result is therefore
  not caused by 100 Hz downsampling.
- On the sensor dataset with 176 training and 39 test groups, mean group R2 is
  `0.425`. This is consistent with the ideal-data learning curve and shows no
  catastrophic sensitivity to the configured gyro/motor sensor noise. It does
  not cover attitude-estimator error.

## Control-Value Audit

Predicted effective labels were converted back into the continuous local model,
discretized at 500 Hz, and used to solve the same DARE as the controller. Each
gain was then evaluated on the true test-group local plant. The reported cost is
the infinite-horizon quadratic cost for initial covariance
`diag(state_scales^2)`.

| Dataset | Test groups | Stable nominal/pred/oracle | Pred cost / nominal | Oracle cost / nominal | Pred gain relative error |
| --- | ---: | ---: | ---: | ---: | ---: |
| medium ideal, `0-1 s` | 75 | 100% / 100% / 100% | 0.882 | 0.839 | 0.175 mean |
| medium sensors, `0-2 s` | 39 | 100% / 100% / 100% | 0.905 | 0.872 | 0.170 mean |

On ideal histories, the predicted gain captures approximately 73% of the mean
cost improvement available to the oracle gain. Mean true-plant pole radius is
`0.9950` for nominal, `0.9933` for predicted, and `0.9926` for oracle.

This is positive evidence of control value, but only for local linearizations of
plants on which the nominal controller eventually converges. It does not prove
that an instantaneous nonlinear gain swap is safe.

## Conclusion

Supervised learning is viable for identifying a local effective LQR model, but
not for uniquely identifying all 13 raw physical scales from this experiment.
The useful output is control effectiveness plus observable actuator dynamics,
with uncertainty, rather than a forced decomposition into mass, inertia, thrust,
and grid coefficients.

The current zero-reference recovery experiment can identify grid control
authority well, motor time constants moderately, and servo time constants
poorly. A deployable system should therefore:

1. predict a constrained local model (or its low-rank latent coordinates) and
   calibrated uncertainty, not all raw physical parameters;
2. fuse many overlapping windows recursively, weighting them by measured
   excitation/information rather than averaging settled flight equally;
3. collect normal multiaxis command maneuvers, throttle changes, and yaw-rate
   changes so actuator dynamics receive persistent excitation; a small bounded
   simultaneous dither can be enabled only when operational motion is
   insufficient;
4. recompute gains only after controllability, parameter-bound, predicted-pole,
   and uncertainty gates pass, then rate-limit/interpolate model or gain updates;
5. retain the nominal/hybrid controller as a fallback and freeze adaptation on
   saturation, estimator faults, low information, or out-of-distribution input;
6. validate with attitude-estimator errors, sensor bias/delay randomization,
   unmodeled aerodynamics, actuator dead zones/rate limits, and nonlinear Monte
   Carlo gain blending before hardware tests.

## Reproduction

Dataset generation, MLP training, target transforms, and local control-value
evaluation are implemented in `Identification/src/flight_identification/`.
Representative commands are documented in `Identification/README.md`. Full
datasets and checkpoints are intentionally excluded from git.

## Joint Plus/Minus Two-Decade Experiment

The deployment-oriented follow-up randomizes all 13 physical labels jointly in
`[-2, 2]` log10 units. The final dataset contains 32,768 parameter groups,
65,536 episodes, and 46,486 safety-valid one-second adaptation windows:
32,645 train, 6,988 validation, and 6,853 test. Every raw parameter reaches
approximately both range endpoints in every split. Splits remain disjoint by
parameter group.

Only 384 complete episodes converged under the fixed nominal LQR. This is an
important feasibility result: a controller that must wait one second before
identification cannot recover trajectories that leave the safety envelope
earlier. The adaptation dataset therefore uses safety-valid prefixes rather
than requiring eventual nominal-controller convergence.

### Identifier Selection

| Candidate | Parameters | Test group mean R2 | Test local stable fraction |
| --- | ---: | ---: | ---: |
| effective MLP, one member | 883k | 0.630-0.639 | 0.618 |
| effective TCN | 837k | 0.630 | 0.651 on v1 |
| five-member effective MLP ensemble | 4.42M | 0.665 | 0.639 ungated |
| direct 20D PCA gain MLP | 884k | n/a | 0.381 |

The direct-gain alternative was rejected. Although 20 PCA coordinates explain
99.91% of training-group gain variance, its test median relative gain error is
0.816 and it is less stable than reconstructing a gain from effective ratios.

The selected ensemble identifies control authority and motor time constants
well enough to improve a broad local audit, but servo time constants remain
weak: their group R2 values are approximately 0.22-0.24. The uncertainty model
is calibrated on individual one-second windows, not averages over multiple
initial conditions.

### Local Gain Gate

The conservative validation-selected policy uses 25% of the predicted gain
step and requires the predicted closed-loop pole radius to be below 0.9999.
On 6,853 held-out test windows it accepts 4,604 (67.2%). Among accepted updates,
99.283% do not turn a nominally stable true plant unstable. It rescues 495
nominally unstable cases and harms 33 nominally stable cases. The resulting
local stable fraction is 0.388, versus 0.320 for the nominal gain. This passes
the configured empirical 99% non-degradation threshold, although its 95%
Wilson lower bound is 0.98995 and should not be interpreted as a hardware
safety guarantee.

### Nonlinear Decision

The final audit uses all 3,490 held-out parameter groups with newly sampled
initial states. Both branches share an exactly identical first nominal-LQR
second (`prefix_max_absolute_difference=0`); accepted gains are interpolated
over two seconds.

| Duration | Accepted | Nominal/adaptive safe, all | Nominal/adaptive converged, accepted | Safety rescued/harmed |
| --- | ---: | ---: | ---: | ---: |
| 4 s | 1,313 | 57.36% / 57.28% | 9.90% / 9.06% | 13 / 10 |
| 8 s | 1,313 | 46.28% / 46.39% | 11.35% / 10.89% | 34 / 36 |

The nonlinear convergence gate therefore fails. The final artifact is frozen
as `deployment_mode=shadow_only` with gain updates disabled. The network has
deployment value for real-time shadow identification, OOD/DARE rejection, and
new data collection, but not yet for automatic LQR scheduling.

The next experiment must add persistent multiaxis excitation during otherwise
normal flight and collect longer overlapping histories. Merely increasing MLP
capacity or regressing oracle gains directly did not solve the missing
information problem.
