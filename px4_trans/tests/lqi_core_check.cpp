/****************************************************************************
 *
 *   Copyright (c) 2026 PX4 Development Team. All rights reserved.
 *
 ****************************************************************************/

#include "lqi_golden.hpp"

#include "../px4/src/modules/nn_control/LqiControllerCore.hpp"

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

	for (int index = 0; index < kLqiGoldenVectorCount; ++index) {
		const LqiGoldenVector &vector = kLqiGoldenVectors[index];
		LqiControllerCore::Input input {};

		for (int i = 0; i < 4; ++i) {
			input.attitude_q[i] = vector.attitude_q[i];
		}

		for (int i = 0; i < 3; ++i) {
			input.angular_velocity[i] = vector.angular_velocity[i];
		}

		input.rc_throttle = vector.rc_throttle;
		input.rc_yaw = vector.rc_yaw;

		LqiControllerCore::ControlOutput output {};
		float next_persistent[LqiControllerCore::kPersistentSize] {};

		const bool valid = LqiControllerCore::stepFromState(
					   input, vector.persistent, output, next_persistent);
		bool ok = valid;
		ok = ok && close_enough(output.upper, vector.expected_upper);
		ok = ok && close_enough(output.lower, vector.expected_output[0]);

		for (int i = 0; i < 3; ++i) {
			ok = ok && close_enough(output.servos[i], vector.expected_output[1 + i]);
		}

		for (int i = 0; i < LqiControllerCore::kPersistentSize; ++i) {
			ok = ok && close_enough(
				     next_persistent[i], vector.expected_next_persistent[i]);
		}

		if (!ok) {
			++failures;
			std::printf("vector %d FAILED (valid=%d)\n", index, valid ? 1 : 0);
			std::printf("  upper got %.9g want %.9g\n",
				    static_cast<double>(output.upper),
				    static_cast<double>(vector.expected_upper));
			std::printf("  lower got %.9g want %.9g\n",
				    static_cast<double>(output.lower),
				    static_cast<double>(vector.expected_output[0]));

			for (int i = 0; i < 3; ++i) {
				std::printf("  servo%d got %.9g want %.9g\n", i,
					    static_cast<double>(output.servos[i]),
					    static_cast<double>(vector.expected_output[1 + i]));
			}

			for (int i = 0; i < LqiControllerCore::kPersistentSize; ++i) {
				std::printf("  persistent%d got %.9g want %.9g\n", i,
					    static_cast<double>(next_persistent[i]),
					    static_cast<double>(vector.expected_next_persistent[i]));
			}
		}
	}

	if (failures == 0) {
		std::printf("LQI core check PASSED (%d vectors)\n", kLqiGoldenVectorCount);
		return 0;
	}

	std::printf("LQI core check FAILED (%d/%d vectors)\n",
		    failures, kLqiGoldenVectorCount);
	return 1;
}
