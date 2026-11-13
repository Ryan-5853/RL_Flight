/****************************************************************************
 *
 *   Copyright (c) 2026 PX4 Development Team. All rights reserved.
 *
 ****************************************************************************/

#pragma once

#include <stdint.h>

/**
 * Weight-independent ABI for the 19-state composite LQI.
 *
 * This is the offline-identified deployment variant (real-log v1).  The state
 * vector is
 *
 *   [roll_error, pitch_error, p, q, r,
 *    upper_f, lower_f,
 *    latent(3 basis x 3 axes), integral(3)]
 *
 * The latent states are command-driven first-order filters in the three
 * orthonormal servo modes produced by ``mode_transform @ servo_slopes``; the
 * identified composite coefficients couple them into the angular-acceleration
 * state.  The 13-state nominal model (LqiControllerModel) is intentionally
 * untouched so the previous weight version remains selectable at runtime.
 */
struct LqiCompositeControllerModel
{
	static constexpr uint32_t kSchemaVersion{2};
	static constexpr int kStateSize{19};
	static constexpr int kOutputSize{4};
	static constexpr int kMotorStateSize{2};
	static constexpr int kLatentStateSize{9};
	static constexpr int kIntegralStateSize{3};
	static constexpr int kPersistentSize{
		kMotorStateSize + kLatentStateSize + kIntegralStateSize
	};

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

	// Row-major K for delta_u = -K x, x in the 19-state order above.
	const float *gain;

	// Upper/lower first-order motor observers (deviations from trim, rad/s).
	const float *motor_decay;
	const float *motor_slope_eff;

	// Latent first-order basis filters. decay[basis] and
	// latent_input[basis][axis][servo] = (T @ diag(slopes))[axis][servo]
	//                                      * (1 - decay[basis]).
	const float *latent_decay;
	const float *latent_input;

	const float *integral_limit;
	const float *motor_command_to_rad_s;
	uint8_t motor_table_points;

	float dt_s;
	float upper_trim;
	float lower_trim;
	float upper_trim_rad_s;
	float lower_trim_rad_s;
	float rpm_to_rad_s;

	// PX4 manual-control adapter (identical convention to the nominal model).
	float max_roll_angle_rad;
	float max_pitch_angle_rad;
	float max_yaw_rate_rad_s;
	const float *manual_command_decay;

	uint32_t weight_bytes;
	uint32_t checksum;
};
