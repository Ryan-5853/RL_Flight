#include "../px4/src/modules/nn_control/LqiCoaxialCollectiveCore.hpp"
#ifndef COAXIAL_PAYLOAD_HEADER
#define COAXIAL_PAYLOAD_HEADER "../px4/src/modules/nn_control/LqiCoaxialCollectivePayload.hpp"
#endif
#include COAXIAL_PAYLOAD_HEADER
#include "../px4/src/modules/nn_control/LqiManualReference.hpp"
#include <cmath>
#include <cstdio>
#include <limits>

using M = LqiCoaxialCollectiveModel;
using Core = LqiCoaxialCollectiveCore;
static int failures = 0;
static void check(bool ok, const char *name) { if (!ok) { ++failures; std::printf("FAIL: %s\n", name); } }
static Core::Input hover()
{
	Core::Input input{};
	input.attitude_q_wb[0] = input.target_attitude_q_wb[0] = 1;
	input.acceleration_b[2] = -9.80665f;
	input.collective_base = .5f;
	return input;
}
static bool step(const M &model, const Core::Input &input, const float *state, Core::ControlOutput &output, float *next)
{
	return Core::stepFromState(model, input, state, output, next);
}
int main()
{
	const M &model = lqi_coaxial_collective::kModel;
	check(model.yaw_balance != nullptr, "active candidate has yaw balance policy");
	if (!model.yaw_balance) { return 1; }
	const float *policy = model.yaw_balance;
	check(std::fabs(model.max_yaw_rate_rad_s - .3f) < 1e-6f, "candidate rate cap");
	float state[5]{}, next[5]{};
	Core::ControlOutput out{};
	auto input = hover();
	check(step(model, input, state, out, next), "hover valid");
	check(out.upper*.5f > .23f && out.upper*.5f < .30f && out.lower*.5f > .33f && out.lower*.5f < .40f, "bounded hover commands with smaller preferred common");
	check(std::fabs((out.servos[0] + out.servos[1] + out.servos[2]) / 3 + .08f) < 1e-5f, "negative grid common bias");

	// Sweep collective, actual/target yaw rate and large prior bias. Every
	// emitted pair, including saturated cases, obeys the speed/command envelope.
	for (int collective = 3; collective <= 100; collective += 3) {
		for (int demand = -6; demand <= 6; ++demand) {
			input = hover(); input.collective_base = collective * .01f;
			input.angular_velocity_b[2] = demand * .3f;
			state[4] = demand * .02f;
			check(step(model, input, state, out, next), "sweep valid");
			const float wu = lqi_coaxial_collective_detail::speed_lookup(out.upper, model.upper_command_to_rad_s, model.upper_motor_table_points);
			const float wl = lqi_coaxial_collective_detail::speed_lookup(out.lower, model.lower_command_to_rad_s, model.lower_motor_table_points);
			check(wu <= policy[M::MaximumSpeedRatio] * wl + .002f && wl <= policy[M::MaximumSpeedRatio] * wu + .002f, "predicted RPM ratio bounded in either direction");
			check(out.upper >= 0 && out.lower >= 0 && out.upper <= .900001f && out.lower <= .900001f, "absolute motor cap");
			check(std::fabs(next[4] * model.attitude_gain[23]) <= .003001f, "yaw integral moment bounded");
		}
	}

	// Hold an impossible yaw-rate error for 20 seconds. No accumulating
	// motor difference or unbounded integral is permitted.
	input = hover(); input.angular_velocity_b[2] = .6f;
	for (float &value : state) { value = 0; }
	for (int i = 0; i < 10000; ++i) {
		check(step(model, input, state, out, next), "persistent error valid");
		for (int j = 0; j < 5; ++j) { state[j] = next[j]; }
	}
	check(std::fabs(state[4] * model.attitude_gain[23]) <= .003001f, "20-second bias remains bounded");
	state[4] = .04f; input.angular_velocity_b[2] = -.01f;
	check(step(model, input, state, out, next) && next[4] < state[4], "integral may unwind while yaw remains saturated");
	input.angular_velocity_b[2] = .1f;
	check(step(model, input, state, out, next) && std::fabs(next[4] - state[4]) < 1e-6f, "integral cannot drive farther into saturation");

	// The core protects direct callers too, not just the stick adapter.
	for (float &value : state) { value = 0; }
	input = hover(); input.target_angular_velocity_b[2] = 100;
	Core::ControlOutput oversized{}, limited{}; float n1[5]{}, n2[5]{};
	check(step(model, input, state, oversized, n1), "oversized reference valid");
	input.target_angular_velocity_b[2] = .3f;
	check(step(model, input, state, limited, n2), "capped reference valid");
	check(std::fabs(oversized.upper-limited.upper) < 1e-6f && std::fabs(oversized.lower-limited.lower) < 1e-6f && std::fabs(n1[4]-n2[4]) < 1e-6f, "core rate reference hard cap");

	// A common bias must give way to roll/pitch without altering their
	// differential vector or clipping a single servo independently.
	input = hover(); input.angular_velocity_b[0] = -2; input.angular_velocity_b[1] = 2;
	float no_common_policy[M::YawBalanceSize];
	for (int i = 0; i < M::YawBalanceSize; ++i) { no_common_policy[i] = policy[i]; }
	no_common_policy[M::GridCommonCommand] = 0;
	M no_common_model = model; no_common_model.yaw_balance = no_common_policy;
	Core::ControlOutput no_common{};
	check(step(model, input, state, out, next) && step(no_common_model, input, state, no_common, next), "roll-pitch priority valid");
	const float common = (out.servos[0] + out.servos[1] + out.servos[2]) / 3;
	check(common > -.159f && common <= 0, "common yields before servo endpoint");
	for (int i = 0; i < 3; ++i) {
		check(std::fabs(out.servos[i]) <= 1.000001f && std::fabs(out.servos[i] - common - no_common.servos[i]) < 1e-5f, "roll-pitch differential retained");
	}

	input = hover(); input.collective_base = 0;
	check(step(model, input, state, out, next) && out.upper == 0 && out.lower == 0 && out.servos[0] == 0 && out.servos[1] == 0 && out.servos[2] == 0, "idle also removes common bias");
	float invalid_policy[M::YawBalanceSize];
	for (int i = 0; i < M::YawBalanceSize; ++i) { invalid_policy[i] = policy[i]; }
	invalid_policy[M::MaximumSpeedRatio] = std::numeric_limits<float>::quiet_NaN();
	M invalid = model; invalid.yaw_balance = invalid_policy;
	check(!step(invalid, hover(), state, out, next), "invalid policy rejected");
	M legacy = model; legacy.yaw_balance = nullptr;
	legacy.collective_contract = M::CollectiveContract::PilotAccelerationDualMotorCommonDifferentialServoRollPitch;
	check(step(legacy, hover(), state, out, next) && std::fabs(out.upper-out.lower) < 1e-5f && out.servos[0] == 0, "null policy retains legacy allocation");

	LqiCoaxialManualReference reference{model}; input = hover();
	float q[4]{1,0,0,0}, rate[3]{}; float previous = 0;
	for (int i = 0; i < 1000; ++i) {
		check(reference.build(q,rate,0,0,1,0,input), "manual reference valid");
		check(input.target_angular_velocity_b[2] >= previous && input.target_angular_velocity_b[2] <= .300001f, "filtered full stick respects rate cap");
		previous = input.target_angular_velocity_b[2];
	}
	std::printf("yaw balance envelope checks: %d failures\n", failures);
	return failures != 0;
}
