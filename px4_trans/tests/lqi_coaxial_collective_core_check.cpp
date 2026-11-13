/****************************************************************************
 *
 *   Copyright (c) 2026 PX4 Development Team. All rights reserved.
 *
 ****************************************************************************/

#ifndef COAXIAL_GOLDEN_HEADER
#define COAXIAL_GOLDEN_HEADER "lqi_coaxial_collective_golden.hpp"
#endif
#include COAXIAL_GOLDEN_HEADER

#include "../px4/src/modules/nn_control/LqiCoaxialCollectiveCore.hpp"
#ifndef COAXIAL_PAYLOAD_HEADER
#define COAXIAL_PAYLOAD_HEADER "../px4/src/modules/nn_control/LqiCoaxialCollectivePayload.hpp"
#endif
#include COAXIAL_PAYLOAD_HEADER
#include "../px4/src/modules/nn_control/LqiManualReference.hpp"

#include <cmath>
#include <cstdio>

namespace
{

bool close_enough(float actual, float expected)
{
	const float tolerance = 1e-4f * (1.0f + std::fabs(expected));
	return std::fabs(actual - expected) <= tolerance;
}

} // namespace

int main()
{
	int failures = 0;

	for (int index = 0; index < kLqiCoaxialCollectiveGoldenVectorCount; ++index) {
		const LqiCoaxialCollectiveGoldenVector &vector =
			kLqiCoaxialCollectiveGoldenVectors[index];
		LqiCoaxialCollectiveCore::Input input {};

		for (int i = 0; i < 4; ++i) {
			input.attitude_q_wb[i] = vector.attitude_q_wb[i];
			input.target_attitude_q_wb[i] = vector.target_attitude_q_wb[i];
		}

		for (int i = 0; i < 3; ++i) {
			input.angular_velocity_b[i] = vector.angular_velocity_b[i];
			input.target_angular_velocity_b[i] =
				vector.target_angular_velocity_b[i];
			input.acceleration_b[i] = vector.acceleration_b[i];
		}

		input.collective_base = vector.collective_base;
		LqiCoaxialCollectiveCore::ControlOutput output {};
		float next_persistent[LqiCoaxialCollectiveCore::kPersistentSize] {};
		const bool valid = LqiCoaxialCollectiveCore::stepFromState(
			lqi_coaxial_collective::kModel,
			input,
			vector.persistent,
			output,
			next_persistent);
		const float outputs[5] {
			output.upper,
			output.lower,
			output.servos[0],
			output.servos[1],
			output.servos[2],
		};
		bool ok = valid;

		for (int i = 0; i < 5; ++i) {
			ok = ok && close_enough(outputs[i], vector.expected_output[i]);
		}

		for (int i = 0; i < LqiCoaxialCollectiveCore::kPersistentSize; ++i) {
			ok = ok && close_enough(
				next_persistent[i], vector.expected_next_persistent[i]);
		}

		if (!ok) {
			++failures;
			std::printf("coaxial vector %d FAILED (valid=%d)\n",
				    index, valid ? 1 : 0);

			for (int i = 0; i < 5; ++i) {
				std::printf("  output%d got %.9g want %.9g\n", i,
					    static_cast<double>(outputs[i]),
					    static_cast<double>(vector.expected_output[i]));
			}

			for (int i = 0; i < LqiCoaxialCollectiveCore::kPersistentSize; ++i) {
				std::printf("  persistent%d got %.9g want %.9g\n", i,
					    static_cast<double>(next_persistent[i]),
					    static_cast<double>(vector.expected_next_persistent[i]));
			}
		}
	}

	// Idle throttle must produce five exact zero outputs.
	{
		LqiCoaxialCollectiveCore::Input input {};
		input.attitude_q_wb[0] = 1.0f;
		input.target_attitude_q_wb[0] = 1.0f;
		const float persistent[LqiCoaxialCollectiveCore::kPersistentSize] {};
		float next_persistent[LqiCoaxialCollectiveCore::kPersistentSize] {};
		LqiCoaxialCollectiveCore::ControlOutput output {};
		const bool ok = LqiCoaxialCollectiveCore::stepFromState(
			lqi_coaxial_collective::kModel,
			input,
			persistent,
			output,
			next_persistent)
			&& output.upper == 0.0f && output.lower == 0.0f
			&& output.servos[0] == 0.0f
			&& output.servos[1] == 0.0f
			&& output.servos[2] == 0.0f;

		if (!ok) {
			++failures;
			std::printf("coaxial idle gate FAILED\n");
		}
	}

	// PX4 positive pitch remains nose-down; both motors remain active after
	// the manual reference is built.
	{
		const float attitude[4] {1.0f, 0.0f, 0.0f, 0.0f};
		const float rate[3] {};
		LqiCoaxialCollectiveCore::Input input {};
		LqiCoaxialManualReference reference{
			lqi_coaxial_collective::kModel};
		const bool built = reference.build(
			attitude, rate, 1.0f, 1.0f, 0.0f, 0.1f, input);
		input.acceleration_b[2] = -9.80665f;
		const float persistent[LqiCoaxialCollectiveCore::kPersistentSize] {};
		float next_persistent[LqiCoaxialCollectiveCore::kPersistentSize] {};
		LqiCoaxialCollectiveCore::ControlOutput output {};
		const bool ok = built
			&& input.target_attitude_q_wb[1] > 0.0f
			&& input.target_attitude_q_wb[2] < 0.0f
			&& close_enough(input.collective_base, 0.55f)
			&& LqiCoaxialCollectiveCore::stepFromState(
				lqi_coaxial_collective::kModel,
				input,
				persistent,
				output,
				next_persistent)
			&& std::isfinite(output.upper)
			&& std::isfinite(output.lower);

		if (!ok) {
			++failures;
			std::printf("coaxial manual-reference contract FAILED\n");
		}
	}

	if (failures == 0) {
		std::printf("coaxial collective LQI core check PASSED (%d vectors)\n",
			    kLqiCoaxialCollectiveGoldenVectorCount);
		return 0;
	}

	std::printf("coaxial collective LQI core check FAILED (%d/%d vectors)\n",
		    failures, kLqiCoaxialCollectiveGoldenVectorCount);
	return 1;
}
