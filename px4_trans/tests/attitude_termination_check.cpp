#include "AttitudeTerminationGuard.hpp"
#include "LqiCoaxialCollectiveCore.hpp"
#include "LqiCoaxialCollectivePayload.hpp"
#include "LqiManualReference.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <limits>

namespace
{
constexpr float kRad = 0.017453292519943295f;

void quaternion(float roll_deg, float pitch_deg, float yaw_deg, float q[4])
{
	const float cr = std::cos(roll_deg * kRad * .5f), sr = std::sin(roll_deg * kRad * .5f);
	const float cp = std::cos(pitch_deg * kRad * .5f), sp = std::sin(pitch_deg * kRad * .5f);
	const float cy = std::cos(yaw_deg * kRad * .5f), sy = std::sin(yaw_deg * kRad * .5f);
	q[0] = cr * cp * cy + sr * sp * sy;
	q[1] = sr * cp * cy - cr * sp * sy;
	q[2] = cr * sp * cy + sr * cp * sy;
	q[3] = cr * cp * sy - sr * sp * cy;
}
}

int main()
{
	float q[4];
	// Both axis signs, arbitrary headings and non-unit quaternion scale.
	for (int axis = 0; axis < 2; ++axis) {
		for (float sign : {-1.f, 1.f}) {
			for (float yaw : {-170.f, 0.f, 170.f}) {
				AttitudeTerminationGuard guard;
				quaternion(axis == 0 ? sign * 29.9f : 0.f, axis == 1 ? sign * 29.9f : 0.f, yaw, q);
				assert(!guard.update(true, true, q));
				quaternion(axis == 0 ? sign * 30.1f : 0.f, axis == 1 ? sign * 30.1f : 0.f, yaw, q);
				for (float &v : q) { v *= -2.f; }
				assert(guard.update(true, true, q));
				quaternion(0.f, 0.f, yaw, q);
				assert(guard.update(true, true, q)); // recovery cannot restart motors
				assert(guard.update(true, false, q)); // switching mode cannot bypass latch
				assert(!guard.update(false, true, q));
				assert(!guard.update(true, true, q));
			}
		}
	}

	AttitudeTerminationGuard guard;
	quaternion(60.f, 0.f, 0.f, q);
	assert(!guard.update(false, true, q));
	assert(!guard.update(true, false, q));
	q[0] = std::numeric_limits<float>::quiet_NaN();
	assert(!guard.update(true, true, q)); // existing invalid-input gate owns this

	// Saturated pilot sticks must command only +/-10 degrees on each axis.
	const float level[4] {1.f, 0.f, 0.f, 0.f};
	const float rates[3] {};
	for (float sign : {-1.f, 1.f}) {
		LqiCoaxialManualReference reference{lqi_coaxial_collective::kModel};
		LqiControllerCore::Input input{};
		assert(reference.build(level, rates, sign * 2.f, sign * 2.f, 0.f, 0.f, input));
		const auto &t = input.target_attitude_q_wb;
		const float roll = std::atan2(2.f * (t[0] * t[1] + t[2] * t[3]),
					     1.f - 2.f * (t[1] * t[1] + t[2] * t[2]));
		const float pitch = std::asin(2.f * (t[0] * t[2] - t[3] * t[1]));
		assert(std::fabs(roll - sign * 10.f * kRad) < 1e-6f);
		assert(std::fabs(pitch + sign * 10.f * kRad) < 1e-6f);
	}

	// Acceleration feedback is world-vertical, including all three body axes.
	// A horizontal acceleration at a tilted attitude must not masquerade as
	// vertical acceleration just because the body-z measurement changes.
	for (float up : {0.f, 2.f}) {
		for (float horizontal : {0.f, 5.f}) {
			LqiControllerCore::Input input{};
			quaternion(20.f, 0.f, 0.f, input.attitude_q_wb);
			for (int i = 0; i < 4; ++i) { input.target_attitude_q_wb[i] = input.attitude_q_wb[i]; }
			const auto &model = lqi_coaxial_collective::kModel;
			const float sr = std::sin(20.f * kRad), cr = std::cos(20.f * kRad);
			const float down_force = -model.gravity_m_s2 - up;
			input.acceleration_b[1] = cr * horizontal + sr * down_force;
			input.acceleration_b[2] = -sr * horizontal + cr * down_force;
			input.collective_base = .5f;
			float persistent[LqiCoaxialCollectiveCore::kPersistentSize] {};
			float next[LqiCoaxialCollectiveCore::kPersistentSize] {};
			LqiControllerCore::ControlOutput output{};
			assert(LqiCoaxialCollectiveCore::stepFromState(model, input, persistent, output, next));
			const float alpha = model.dt_s / (model.acceleration_filter_time_constant_s + model.dt_s);
			assert(std::fabs(next[0] - alpha * up) < 2e-6f);
		}
	}

	std::puts("30 deg termination, 10 deg reference and world-vertical feedback checks passed");
}
