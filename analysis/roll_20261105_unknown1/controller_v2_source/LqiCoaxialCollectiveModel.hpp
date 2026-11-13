/****************************************************************************
 *
 *   Copyright (c) 2026 PX4 Development Team. All rights reserved.
 *
 ****************************************************************************/

#pragma once

#include <stdint.h>

/**
 * Weight-independent ABI for the coaxial collective/servo LQI.
 *
 * The pilot throttle requests a vertical acceleration. A z-acceleration PI
 * controls the coaxial common mode. The 8-state attitude LQI produces three
 * desired moments. The legacy allocator uses motor differential for yaw and
 * servos for roll/pitch. An optional deployment envelope adds bounded grid
 * common-mode yaw trim, near-equal predicted rotor speeds, and anti-windup.
 */
struct LqiCoaxialCollectiveModel
{
	static constexpr uint32_t kSchemaVersion{5};
	static constexpr int kStateSize{8};
	static constexpr int kAttitudeInputSize{3};
	static constexpr int kOutputSize{5};
	static constexpr int kPersistentSize{5};
	static constexpr int kMotorCount{2};
	static constexpr int kServoCount{3};

	enum class WorldFrame : uint8_t { NED = 1 };
	enum class BodyFrame : uint8_t { FRD = 1 };
	enum class QuaternionConvention : uint8_t { HamiltonWxyzBodyToWorld = 1 };
	enum class CollectiveContract : uint8_t {
		PilotAccelerationDualMotorCommonDifferentialServoRollPitch = 2,
		PilotAccelerationBoundedRotorDifferentialGridYawTrim = 3,
		PilotAccelerationSharedYawAntiWindup = 4
	};

	uint32_t schema_version;
	WorldFrame world_frame;
	BodyFrame body_frame;
	QuaternionConvention quaternion_convention;
	CollectiveContract collective_contract;

	// Row-major K for desired moments = -Kx.
	const float *attitude_gain;

	// [servo_1, servo_2, servo_3] = servo_direction * [roll, pitch] / scale.
	const float *servo_direction;
	const float *maximum_moment_n_m;

	// Contract-command-domain local [dT,dMz]/d[upper,lower] lookup.
	// motor_effectiveness has four scalars per point:
	// [dT/d_u, dT/d_l, dMz/d_u, dMz/d_l].
	const float *motor_effectiveness_points;
	const float *motor_effectiveness;
	uint8_t motor_effectiveness_points_count;

	const float *upper_command_to_rad_s;
	const float *lower_command_to_rad_s;
	uint8_t upper_motor_table_points;
	uint8_t lower_motor_table_points;

	float mass_kg;
	float gravity_m_s2;
	float thrust_coefficients[3];
	float motor_torque_coefficients[2];
	float maximum_thrust_n;
	float minimum_thrust_fraction;
	float maximum_thrust_fraction;
	float tilt_cosine_minimum;

	float pilot_minimum;
	float pilot_maximum;
	float maximum_vertical_acceleration_m_s2;
	float acceleration_kp;
	float acceleration_ki;
	float acceleration_integral_limit_m_s2;
	float acceleration_filter_time_constant_s;

	float servo_effectiveness_scale;
	float servo_effectiveness_scale_minimum;
	float servo_effectiveness_scale_maximum;
	float motor_mix_regularization;
	float zero_throttle_threshold;

	float dt_s;
	float max_roll_angle_rad;
	float max_pitch_angle_rad;
	float max_yaw_rate_rad_s;
	const float *manual_command_decay;

	uint32_t weight_bytes;
	uint32_t checksum;

	// Optional deployment envelope. Null preserves the legacy allocation.
	// Commands are PRE x0.5 map; ratios constrain model-predicted steady RPM,
	// not measured ESC speed. The lower torque scale is an experimental model
	// correction, not a newly identified aerodynamic coefficient.
	enum YawBalanceIndex {
		TrimSpeedRatio, MaximumSpeedRatio, MotorCommandMaximum,
		GridCommonCommand, GridYawPerThrustPerCommand,
		YawIntegralMomentLimit, LowerTorqueScale, YawBalanceSize
	};
	const float *yaw_balance{nullptr};

	// Contract 4: reserve P/D authority by limiting each roll/pitch integral
	// moment to a fraction of that axis's moment cap. Preferred grid common
	// is a fraction of the existing signed maximum and may move within [max,0].
	float roll_pitch_integral_moment_fraction{1.0f};
	float grid_common_preference_fraction{1.0f};
};
