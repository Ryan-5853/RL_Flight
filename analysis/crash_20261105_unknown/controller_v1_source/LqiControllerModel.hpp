/****************************************************************************
 *
 *   Copyright (c) 2026 PX4 Development Team. All rights reserved.
 *
 ****************************************************************************/

#pragma once

#include <stdint.h>

/**
 * Weight-independent ABI consumed by the LQI execution framework.
 *
 * A generated model header owns the numerical payload and constructs one of
 * these descriptors.  Changing gains therefore does not change the control
 * algorithm or its PX4-facing interface.
 */
struct LqiControllerModel {
	static constexpr uint32_t kSchemaVersion{1};
	static constexpr int kStateSize{13};
	static constexpr int kOutputSize{4};
	static constexpr int kActuatorStateSize{5};
	static constexpr int kIntegralStateSize{3};
	static constexpr int kPersistentSize{kActuatorStateSize + kIntegralStateSize};

	enum class WorldFrame : uint8_t { NED = 1 };
	enum class BodyFrame : uint8_t { FRD = 1 };
	enum class QuaternionConvention : uint8_t { HamiltonWxyzBodyToWorld = 1 };
	enum class CollectiveContract : uint8_t { ExternalUpperLqiLowerAndServos = 1 };
	enum class ActuatorObserver : uint8_t { MotorRpmOrCommandDriven = 1 };

	uint32_t schema_version;
	WorldFrame world_frame;
	BodyFrame body_frame;
	QuaternionConvention quaternion_convention;
	CollectiveContract collective_contract;
	ActuatorObserver actuator_observer;

	// Row-major K for delta_u = -K x.
	const float *gain;
	const float *decay;
	const float *slope_eff;
	const float *integral_limit;
	const float *motor_command_to_rad_s;
	uint8_t motor_table_points;

	float dt_s;
	float upper_trim;
	float lower_trim;
	float upper_trim_rad_s;
	float lower_trim_rad_s;
	float rpm_to_rad_s;

	// PX4 manual-control adapter. Positive PX4 pitch stick means nose down,
	// hence its sign is handled by the adapter rather than hidden here.
	float max_roll_angle_rad;
	float max_pitch_angle_rad;
	float max_yaw_rate_rad_s;
	const float *manual_command_decay;

	uint32_t weight_bytes;
	uint32_t checksum;
};
