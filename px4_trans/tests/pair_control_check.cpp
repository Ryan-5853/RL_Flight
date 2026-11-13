#include <initializer_list>
#include "../px4/src/modules/comp_control/PairWrench.hpp"
#include "../px4/src/modules/comp_control/PairControlCore.hpp"
#include "../px4/src/modules/nn_control/LqiCoaxialCollectivePayload.hpp"
#include <cassert>
#include <cstdio>
int main(int argc, char **)
{
	if (argc > 1) {
		for (int axis = -1; axis < 3; ++axis) {
			PairControlCore core(lqi_coaxial_collective::kModel);
			PairControlCore::Input in{};
			PairControlCore::Output out;
			in.attitude_q_wb[0] = in.target_attitude_q_wb[0] = 1;
			in.acceleration_b[2] = -lqi_coaxial_collective::kModel.gravity_m_s2;
			in.collective_base = .5f;

			if (axis >= 0) {
				in.target_angular_velocity_b[axis] = .1f;
			}

			assert(core.step(in, .002f, out));
			printf("%d", axis);

			for (float v : out.requested) {
				printf(" %.10g", double(v));
			}

			for (const auto *cmd : {
			             &out.a, &out.b
			     }) {
				printf(" %.10g %.10g", double(cmd->upper), double(cmd->lower));
				for (float v : cmd->servos) printf(" %.10g", double(v));
			}
			float achieved[4] {};
			pair_plant::wrench(lqi_coaxial_collective::kModel, out.a, out.b, achieved);
			for (float v : achieved) printf(" %.10g", double(v));
			puts("");
		}

		return 0;
	}

	PairControlCore core(lqi_coaxial_collective::kModel);
	PairControlCore::Input in{};
	in.attitude_q_wb[0] = in.target_attitude_q_wb[0] = 1;
	in.acceleration_b[2] = -lqi_coaxial_collective::kModel.gravity_m_s2;
	in.collective_base = .5f;
	PairControlCore::Output out;
	assert(core.step(in, .002f, out));
	assert(fabsf(out.requested[0] - 2 * lqi_coaxial_collective::kModel.mass_kg *
	             lqi_coaxial_collective::kModel.gravity_m_s2) < 1e-4f);
	in.target_angular_velocity_b[0] = .02f;
	assert(core.step(in, .002f, out));
	assert(out.side_thrust[1] > out.side_thrust[0]);
	assert(fabsf(out.side_thrust[0] + out.side_thrust[1] - out.requested[0]) < 1e-5f);
	in.target_angular_velocity_b[1] = .02f;
	assert(core.step(in, .002f, out));
	bool different = false;

	for (int i = 0; i < 3; ++i) {
		different |= fabsf(out.a.servos[i] - out.b.servos[i]) > 1e-5f;
	}

	assert(different);
	in.collective_base = 0;
	assert(core.step(in, .002f, out));
	assert(out.a.upper == 0 && out.b.lower == 0);
	in.collective_base = .5f;
	in.acceleration_b[0] = NAN;
	assert(!core.step(in, .002f, out));
	in.acceleration_b[0] = 0;
	assert(!core.step(in, .02f, out));
	// Saturation scales complete vectors and keeps all ten commands in contract.
	in.target_angular_velocity_b[0] = 100.f;
	in.target_angular_velocity_b[1] = 100.f;
	in.target_angular_velocity_b[2] = 100.f;
	core.reset();
	assert(core.step(in, .002f, out));
	assert(out.servo_scale < 1.f && out.yaw_scale[0] < 1.f);
	assert(out.side_thrust[0] > 0 && out.side_thrust[1] > 0);
	for (const auto *cmd : {
	             &out.a, &out.b
	     }) {
		assert(cmd->upper >= 0.f && cmd->upper <= 1.f && cmd->lower >= 0.f && cmd->lower <= 1.f);
		for (float v : cmd->servos) assert(fabsf(v) <= 1.f);
	}
// Match the single-aircraft Pitch angular acceleration (no extra /2).
	assert(fabsf(pair_gains::gain[9] - 2 * lqi_coaxial_collective::kModel.attitude_gain[9]) < 1e-5f);
// Isolated C-axis perturbations expose feedback sign and unintended cross-axis allocation.
	for (int axis = 0; axis < 2; ++axis) for (float sign : {-1.f, 1.f}) {
		core.reset();
		in = {};
		in.attitude_q_wb[0] = cosf(.05f);
		in.attitude_q_wb[axis + 1] = sign * sinf(.05f);
		in.target_attitude_q_wb[0] = 1.f;
		in.acceleration_b[2] = -lqi_coaxial_collective::kModel.gravity_m_s2;
		in.collective_base = .5f;
		assert(core.step(in, .002f, out));
		assert(out.trace_flags == 1u);
		assert(fabsf(out.attitude_error[axis] - sign * .1f) < 1e-6f);
		assert(fabsf(out.attitude_error[1-axis]) < 1e-6f);
		assert(out.requested[axis + 1] * sign < 0.f);
		for (int i = 0; i < 3; ++i) {
			assert(fabsf(out.moment_attitude[i] + out.moment_rate[i] + out.moment_integral[i] - out.requested[i+1]) < 1e-6f);
		}
		assert(fabsf(out.corrected_vertical_acceleration - out.acceleration_pi_p - out.acceleration_pi_i) < 1e-6f);
		if (axis == 1) assert(fabsf(out.side_thrust[0] - out.side_thrust[1]) < 1e-6f);
		in.collective_base = 0.f;
		assert(core.step(in, .002f, out));
		assert(out.trace_flags == 3u && out.a.upper == 0.f && out.b.upper == 0.f);
	}
	// Nonlinear steady-plant small-step closed-loop checks, 6 seconds each.
	for (int scenario = 0; scenario < 5; ++scenario) {
		core.reset();
		in = {};
		float q[4] {1, 0, 0, 0}, rates[3] {};
		for (int tick = 0; tick < 3000; ++tick) {
			for (int i = 0; i < 4; ++i) in.attitude_q_wb[i] = q[i];
			in.target_attitude_q_wb[0] = 1;
			if (scenario == 1 || scenario == 4) {
				in.target_attitude_q_wb[1] = .01f;
			}
			if (scenario == 2 || scenario == 4) {
				in.target_attitude_q_wb[2] = .01f;
			}
			float norm = 0;
			for (float v : in.target_attitude_q_wb) norm += v * v;
			for (float &v : in.target_attitude_q_wb) v /= sqrtf(norm);
			if (scenario == 3 || scenario == 4) in.target_angular_velocity_b[2] = .03f;
			for (int i = 0; i < 3; ++i) in.angular_velocity_b[i] = rates[i];
			const float g = lqi_coaxial_collective::kModel.gravity_m_s2;
			in.acceleration_b[0] = -g * 2 * (q[1] * q[3] - q[0] * q[2]);
			in.acceleration_b[1] = -g * 2 * (q[2] * q[3] + q[0] * q[1]);
			in.acceleration_b[2] = -g * (1 - 2 * (q[1] * q[1] + q[2] * q[2]));
			in.collective_base = .5f;
			assert(core.step(in, .002f, out));
			float wrench[4] {};
			pair_plant::wrench(lqi_coaxial_collective::kModel, out.a, out.b, wrench);
			float angular_momentum[3] {};
			for (int i = 0; i < 3; ++i) angular_momentum[i] = pair_gains::inertia[i] * rates[i];
			float gyro[3] {};
			pair_plant::cross(rates, angular_momentum, gyro);
			for (int i = 0; i < 3; ++i) {
				rates[i] += .002f * (wrench[i + 1] - gyro[i]) / pair_gains::inertia[i];
				assert(fabsf(rates[i]) < .5f);
			}
			float dq[4] {-q[1] *rates[0] - q[2] *rates[1] - q[3] *rates[2], q[0] *rates[0] + q[2] *rates[2] - q[3] *rates[1], q[0] *rates[1] + q[3] *rates[0] - q[1] *rates[2], q[0] *rates[2] + q[1] *rates[1] - q[2] *rates[0]};
			norm = 0;
			for (int i = 0; i < 4; ++i) {
				q[i] += .001f * dq[i];
				norm += q[i] * q[i];
			}
			for (float &v : q) v /= sqrtf(norm);
			if (tick > 2000) {
				assert(out.servo_scale > .99f);
				assert(out.yaw_scale[0] > .99f && out.yaw_scale[1] > .99f);
			}
		}
		assert(fabsf(rates[0]) < .01f && fabsf(rates[1]) < .01f);
		assert(fabsf(rates[2] - in.target_angular_velocity_b[2]) < .01f);
	}
	puts("pair geometry, feedback, allocation, invalid/timing and nonlinear closed-loop checks passed");
}
