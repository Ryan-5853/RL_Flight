/****************************************************************************
 *
 *   Copyright (c) 2026 PX4 Development Team. All rights reserved.
 *
 ****************************************************************************/

#include "lqi_golden.hpp"

#include "../px4/src/modules/nn_control/LqiControllerCore.hpp"
#include "../px4/src/modules/nn_control/LqiManualReference.hpp"
#include "../px4/src/modules/nn_control/LqiNominalModel.hpp"

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
			input.attitude_q_wb[i] = vector.attitude_q_wb[i];
			input.target_attitude_q_wb[i] = vector.target_attitude_q_wb[i];
		}

		for (int i = 0; i < 3; ++i) {
			input.angular_velocity_b[i] = vector.angular_velocity_b[i];
			input.target_angular_velocity_b[i] = vector.target_angular_velocity_b[i];
		}

		input.collective_base = vector.collective_base;
		input.motor_rpm[0] = vector.motor_rpm[0];
		input.motor_rpm[1] = vector.motor_rpm[1];
		input.motor_rpm_valid = vector.motor_rpm_valid;

		LqiControllerCore::ControlOutput output {};
		float next_persistent[LqiControllerCore::kPersistentSize] {};

		const bool valid = LqiControllerCore::stepFromState(
					   lqi_nominal::kModel, input, vector.persistent, output, next_persistent);
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

	// Reference-frame contract: q and -q must be equivalent.
	{
		LqiControllerCore::Input positive {};
		LqiControllerCore::Input negative {};
		const float attitude[4] {0.9238795f, 0.0f, 0.0f, 0.3826834f};
		const float rate[3] {0.0f, 0.0f, 0.0f};
		LqiManualReference positive_reference{lqi_nominal::kModel};
		LqiManualReference negative_reference{lqi_nominal::kModel};
		bool ok = positive_reference.build(attitude, rate, 0.4f, 0.3f, -0.2f, 0.0f, positive);
		const float negative_attitude[4] {-attitude[0], -attitude[1], -attitude[2], -attitude[3]};
		ok = ok && negative_reference.build(negative_attitude, rate,
						       0.4f, 0.3f, -0.2f, 0.0f, negative);
		float state[8] {};
		float next_positive[8] {};
		float next_negative[8] {};
		LqiControllerCore::ControlOutput output_positive {};
		LqiControllerCore::ControlOutput output_negative {};
		ok = ok && LqiControllerCore::stepFromState(lqi_nominal::kModel, positive, state,
							       output_positive, next_positive);
		ok = ok && LqiControllerCore::stepFromState(lqi_nominal::kModel, negative, state,
							       output_negative, next_negative);
		ok = ok && close_enough(output_positive.upper, output_negative.upper)
		     && close_enough(output_positive.lower, output_negative.lower);

		for (int i = 0; i < 3; ++i) {
			ok = ok && close_enough(output_positive.servos[i], output_negative.servos[i]);
		}

		if (!ok) {
			++failures;
			std::printf("reference-frame q/-q invariance FAILED\n");
		}
	}

	// PX4 positive pitch stick is nose-down: target quaternion y must be
	// negative, while positive roll remains positive x. Upper is passthrough.
	{
		const float attitude[4] {1.0f, 0.0f, 0.0f, 0.0f};
		const float rate[3] {};
		LqiControllerCore::Input input {};
		LqiManualReference reference{lqi_nominal::kModel};
		const bool ok = reference.build(attitude, rate, 1.0f, 1.0f, 0.0f, -0.25f, input)
				&& input.target_attitude_q_wb[1] > 0.0f
				&& input.target_attitude_q_wb[2] < 0.0f
				&& close_enough(input.collective_base, 0.375f);

		if (!ok) {
			++failures;
			std::printf("PX4 manual reference sign contract FAILED\n");
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
