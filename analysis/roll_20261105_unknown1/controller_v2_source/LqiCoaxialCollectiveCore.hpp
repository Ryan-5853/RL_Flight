/****************************************************************************
 *
 *   Copyright (c) 2026 PX4 Development Team. All rights reserved.
 *
 ****************************************************************************/

#pragma once

#include "LqiCoaxialCollectiveModel.hpp"
#include "LqiControllerCore.hpp"

#include <cmath>

namespace lqi_coaxial_collective_detail
{

inline float clamp(float value, float lower, float upper)
{
	return (value < lower) ? lower : ((value > upper) ? upper : value);
}

inline float inverse_lookup(
	float value, const float *table, int point_count)
{
	int upper = 1;

	while (upper < point_count - 1 && value > table[2 * upper + 1]) {
		++upper;
	}

	const int lower = upper - 1;
	const float command_0 = table[2 * lower];
	const float speed_0 = table[2 * lower + 1];
	const float command_1 = table[2 * upper];
	const float speed_1 = table[2 * upper + 1];

	if (speed_1 <= speed_0) {
		return command_1;
	}

	if (value <= table[1]) {
		return table[2];
	}

	if (value >= table[2 * (point_count - 1) + 1]) {
		return table[2 * (point_count - 1)];
	}

	return command_0
	       + (value - speed_0) * (command_1 - command_0)
	       / (speed_1 - speed_0);
}

inline float speed_lookup(float command, const float *table, int point_count)
{
	int upper = 1;

	while (upper < point_count - 1 && command > table[2 * upper]) { ++upper; }

	const int lower = upper - 1;
	const float width = table[2 * upper] - table[2 * lower];
	const float fraction = width > 0.0f ? clamp((command - table[2 * lower]) / width, 0.0f, 1.0f) : 0.0f;
	return table[2 * lower + 1] + fraction * (table[2 * upper + 1] - table[2 * lower + 1]);
}

inline bool feasible_motors(const LqiCoaxialCollectiveModel &model, float upper, float lower)
{
	using M = LqiCoaxialCollectiveModel;
	const float *policy = model.yaw_balance;
	const float cap = policy[M::MotorCommandMaximum];

	if (upper < 0.0f || lower < 0.0f || upper > cap || lower > cap) { return false; }

	const float wu = speed_lookup(upper, model.upper_command_to_rad_s, model.upper_motor_table_points);
	const float wl = speed_lookup(lower, model.lower_command_to_rad_s, model.lower_motor_table_points);
	const float ratio = policy[M::MaximumSpeedRatio];
	return wu <= ratio * wl + 1e-3f && wl <= ratio * wu + 1e-3f;
}

inline float allocate_bounded_motors(const LqiCoaxialCollectiveModel &model,
	float upper_base, float lower_base, float upper_delta, float lower_delta,
	float &upper, float &lower)
{
	float fraction = 1.0f;

	if (!feasible_motors(model, upper_base + upper_delta, lower_base + lower_delta)) {
		float low = 0.0f;
		float high = 1.0f;

		for (int iteration = 0; iteration < 16; ++iteration) {
			const float middle = 0.5f * (low + high);

			if (feasible_motors(model, upper_base + middle * upper_delta, lower_base + middle * lower_delta)) { low = middle; }
			else { high = middle; }
		}

		fraction = low;
	}

	upper = upper_base + fraction * upper_delta;
	lower = lower_base + fraction * lower_delta;
	return fraction;
}

inline void interpolate_vector(
	const float *points,
	const float *values,
	int point_count,
	int value_count,
	float point,
	float *output)
{
	int upper = 1;

	while (upper < point_count - 1 && point > points[upper]) {
		++upper;
	}

	const int lower = upper - 1;
	const float width = points[upper] - points[lower];
	const float fraction =
		width > 0.0f ? clamp((point - points[lower]) / width, 0.0f, 1.0f) : 0.0f;

	for (int index = 0; index < value_count; ++index) {
		output[index] =
			values[value_count * lower + index]
			+ fraction
			* (values[value_count * upper + index]
			   - values[value_count * lower + index]);
	}
}

} // namespace lqi_coaxial_collective_detail

/**
 * Host-compilable coaxial collective/servo LQI core.
 *
 * Persistent state (5):
 *   [filtered_vertical_acceleration, vertical_acceleration_integral,
 *    integral_roll_error, integral_pitch_error, integral_yaw_rate_error]
 *
 * Both motor outputs are active. The pilot collective requests a vertical
 * acceleration and the program computes motor common/differential corrections.
 * The optional yaw-balance envelope adds a roll/pitch-priority grid common trim.
 */
class LqiCoaxialCollectiveCore
{
public:
	using Input = LqiControllerCore::Input;
	using ControlOutput = LqiControllerCore::ControlOutput;

	static constexpr int kPersistentSize =
		LqiCoaxialCollectiveModel::kPersistentSize;

	explicit LqiCoaxialCollectiveCore(
		const LqiCoaxialCollectiveModel &model) : _model(model) {}

	void reset();
	bool step(const Input &input, ControlOutput &output);

	static bool stepFromState(
		const LqiCoaxialCollectiveModel &model,
		const Input &input,
		const float persistent[kPersistentSize],
		ControlOutput &output,
		float next_persistent[kPersistentSize]);

	const float *persistent() const { return _persistent; }
	const ControlOutput &lastOutput() const { return _last_output; }

private:
	const LqiCoaxialCollectiveModel &_model;
	float _persistent[kPersistentSize] {};
	ControlOutput _last_output {};
};

inline void LqiCoaxialCollectiveCore::reset()
{
	for (float &value : _persistent) {
		value = 0.0f;
	}

	for (float &value : _last_output.servos) {
		value = 0.0f;
	}

	_last_output.upper = 0.0f;
	_last_output.lower = 0.0f;
}

inline bool LqiCoaxialCollectiveCore::stepFromState(
	const LqiCoaxialCollectiveModel &model,
	const Input &input,
	const float persistent[kPersistentSize],
	ControlOutput &output,
	float next_persistent[kPersistentSize])
{
	if (model.schema_version != LqiCoaxialCollectiveModel::kSchemaVersion
	    || model.world_frame != LqiCoaxialCollectiveModel::WorldFrame::NED
	    || model.body_frame != LqiCoaxialCollectiveModel::BodyFrame::FRD
	    || model.quaternion_convention
	       != LqiCoaxialCollectiveModel::QuaternionConvention::HamiltonWxyzBodyToWorld
	    || (model.collective_contract
		!= LqiCoaxialCollectiveModel::CollectiveContract::PilotAccelerationDualMotorCommonDifferentialServoRollPitch
		&& model.collective_contract
		!= LqiCoaxialCollectiveModel::CollectiveContract::PilotAccelerationBoundedRotorDifferentialGridYawTrim
		&& model.collective_contract
		!= LqiCoaxialCollectiveModel::CollectiveContract::PilotAccelerationSharedYawAntiWindup)
	    || model.attitude_gain == nullptr || model.servo_direction == nullptr
	    || model.maximum_moment_n_m == nullptr
	    || model.motor_effectiveness_points == nullptr
	    || model.motor_effectiveness == nullptr
	    || model.motor_effectiveness_points_count < 2
	    || model.upper_command_to_rad_s == nullptr
	    || model.lower_command_to_rad_s == nullptr
	    || model.upper_motor_table_points < 2
	    || model.lower_motor_table_points < 2
	    || model.manual_command_decay == nullptr
	    || !(model.dt_s > 0.0f) || !(model.mass_kg > 0.0f)
	    || !(model.gravity_m_s2 > 0.0f) || !(model.maximum_thrust_n > 0.0f)) {
		return false;
	}

	using M = LqiCoaxialCollectiveModel;
	const float *policy = model.yaw_balance;

	const bool shared_yaw = model.collective_contract == M::CollectiveContract::PilotAccelerationSharedYawAntiWindup;

	if ((policy != nullptr) != (shared_yaw || model.collective_contract == M::CollectiveContract::PilotAccelerationBoundedRotorDifferentialGridYawTrim)) {
		return false;
	}

	if (policy != nullptr) {
		for (int i = 0; i < M::YawBalanceSize; ++i) {
			if (!std::isfinite(policy[i])) { return false; }
		}

		if (policy[M::TrimSpeedRatio] < 1.0f
		    || policy[M::MaximumSpeedRatio] < policy[M::TrimSpeedRatio]
		    || policy[M::MotorCommandMaximum] <= 0.0f || policy[M::MotorCommandMaximum] > 1.0f
		    || fabsf(policy[M::GridCommonCommand]) > 1.0f
		    || policy[M::YawIntegralMomentLimit] <= 0.0f
		    || policy[M::LowerTorqueScale] <= 0.0f) { return false; }
	}

	if (shared_yaw && (!std::isfinite(model.roll_pitch_integral_moment_fraction)
	    || !(model.roll_pitch_integral_moment_fraction > 0.0f) || model.roll_pitch_integral_moment_fraction > 1.0f
	    || !std::isfinite(model.grid_common_preference_fraction)
	    || model.grid_common_preference_fraction < 0.0f || model.grid_common_preference_fraction > 1.0f)) {
		return false;
	}

	const float tw = input.target_attitude_q_wb[0];
	const float tx = -input.target_attitude_q_wb[1];
	const float ty = -input.target_attitude_q_wb[2];
	const float tz = -input.target_attitude_q_wb[3];
	const float cw = input.attitude_q_wb[0];
	const float cx = input.attitude_q_wb[1];
	const float cy = input.attitude_q_wb[2];
	const float cz = input.attitude_q_wb[3];
	float w = tw * cw - tx * cx - ty * cy - tz * cz;
	float vx = tw * cx + tx * cw + ty * cz - tz * cy;
	float vy = tw * cy - tx * cz + ty * cw + tz * cx;
	float vz = tw * cz + tx * cy - ty * cx + tz * cw;

	if (w < 0.0f) {
		w = -w;
		vx = -vx;
		vy = -vy;
		vz = -vz;
	}

	const float vnorm = std::sqrt(vx * vx + vy * vy + vz * vz);
	const float vnorm_clamped = (vnorm < 1e-12f) ? 1e-12f : vnorm;
	const float angle = 2.0f * std::atan2(vnorm, (w < 1e-12f) ? 1e-12f : w);
	const float scale = angle / vnorm_clamped;
	const float roll_error = vx * scale;
	const float pitch_error = vy * scale;
	const float rate_error[3] {
		input.angular_velocity_b[0] - input.target_angular_velocity_b[0],
		input.angular_velocity_b[1] - input.target_angular_velocity_b[1],
		input.angular_velocity_b[2] - (policy != nullptr
			? lqi_coaxial_collective_detail::clamp(input.target_angular_velocity_b[2],
				-model.max_yaw_rate_rad_s, model.max_yaw_rate_rad_s)
			: input.target_angular_velocity_b[2]),
	};

	using lqi_coaxial_collective_detail::clamp;
	using lqi_coaxial_collective_detail::inverse_lookup;
	using lqi_coaxial_collective_detail::interpolate_vector;
	const float collective = clamp(input.collective_base, 0.0f, 1.0f);
	const bool zero_throttle = collective <= model.zero_throttle_threshold;

	if (zero_throttle) {
		output.upper = 0.0f;
		output.lower = 0.0f;
		output.servos[0] = 0.0f;
		output.servos[1] = 0.0f;
		output.servos[2] = 0.0f;
		next_persistent[0] = persistent[0];

		for (int index = 1; index < kPersistentSize; ++index) {
			next_persistent[index] = 0.0f;
		}

		return true;
	}

	const float qw = input.attitude_q_wb[0];
	const float qx = input.attitude_q_wb[1];
	const float qy = input.attitude_q_wb[2];
	const float qz = input.attitude_q_wb[3];
	const float rotation_31 = 2.0f * (qx * qz - qw * qy);
	const float rotation_32 = 2.0f * (qy * qz + qw * qx);
	const float rotation_33 = 1.0f - 2.0f * (qx * qx + qy * qy);
	const float specific_force_z_ned =
		rotation_31 * input.acceleration_b[0]
		+ rotation_32 * input.acceleration_b[1]
		+ rotation_33 * input.acceleration_b[2];
	const float measured_up_acceleration =
		-(specific_force_z_ned + model.gravity_m_s2);
	const float filter_alpha =
		model.acceleration_filter_time_constant_s > 0.0f
		? model.dt_s
		/ (model.acceleration_filter_time_constant_s + model.dt_s)
		: 1.0f;
	const float filtered_up_acceleration =
		persistent[0]
		+ filter_alpha * (measured_up_acceleration - persistent[0]);
	const float normalized_collective =
		2.0f * clamp(
			(collective - model.pilot_minimum)
			/ (model.pilot_maximum - model.pilot_minimum),
			0.0f, 1.0f)
		- 1.0f;
	const float target_up_acceleration =
		normalized_collective * model.maximum_vertical_acceleration_m_s2;
	const float acceleration_error =
		target_up_acceleration - filtered_up_acceleration;
	const float candidate_acceleration_integral =
		clamp(
			persistent[1] + acceleration_error * model.dt_s,
			-model.acceleration_integral_limit_m_s2,
			model.acceleration_integral_limit_m_s2);
	const float corrected_up_acceleration =
		target_up_acceleration
		+ model.acceleration_kp * acceleration_error
		+ model.acceleration_ki * candidate_acceleration_integral;
	const float acceleration_limit =
		1.5f * model.maximum_vertical_acceleration_m_s2
		+ model.acceleration_integral_limit_m_s2;
	const float bounded_up_acceleration =
		clamp(
			corrected_up_acceleration,
			-model.gravity_m_s2 - acceleration_limit,
			acceleration_limit);
	const float tilt_cosine =
		fmaxf(1.0f - 2.0f * (qx * qx + qy * qy), model.tilt_cosine_minimum);
	const float desired_thrust =
		clamp(
			model.mass_kg * (model.gravity_m_s2 + bounded_up_acceleration)
			/ tilt_cosine,
			model.minimum_thrust_fraction * model.maximum_thrust_n,
			model.maximum_thrust_fraction * model.maximum_thrust_n);

	// Balanced coaxial targets: torque coefficients define the lower/upper
	// speed ratio, then each rotor uses its own command-domain speed table.
	const float speed_ratio = policy != nullptr ? policy[M::TrimSpeedRatio] :
		std::sqrt(
			model.motor_torque_coefficients[0]
			/ fmaxf(model.motor_torque_coefficients[1], 1e-16f));
	const float thrust_denominator =
		model.thrust_coefficients[0]
		+ model.thrust_coefficients[1] * speed_ratio * speed_ratio
		+ model.thrust_coefficients[2] * speed_ratio;
	float upper_target_speed =
		std::sqrt(fmaxf(desired_thrust, 0.0f) / fmaxf(thrust_denominator, 1e-16f));
	bool base_saturated = false;

	if (policy != nullptr) {
		const float upper_cap = lqi_coaxial_collective_detail::speed_lookup(
			policy[M::MotorCommandMaximum], model.upper_command_to_rad_s, model.upper_motor_table_points);
		const float lower_cap = lqi_coaxial_collective_detail::speed_lookup(
			policy[M::MotorCommandMaximum], model.lower_command_to_rad_s, model.lower_motor_table_points);
		const float speed_cap = fminf(upper_cap, lower_cap / speed_ratio);
		base_saturated = upper_target_speed > speed_cap;
		upper_target_speed = fminf(upper_target_speed, speed_cap);
	}

	const float lower_target_speed = speed_ratio * upper_target_speed;
	const float upper_base = inverse_lookup(
		upper_target_speed, model.upper_command_to_rad_s,
		model.upper_motor_table_points);
	const float lower_base = inverse_lookup(
		lower_target_speed, model.lower_command_to_rad_s,
		model.lower_motor_table_points);

	const float base_thrust = thrust_denominator * upper_target_speed * upper_target_speed;
	const float errors[3] {roll_error, pitch_error, rate_error[2]};
	float attitude_integral[3] {};

	for (int axis = 0; axis < 3; ++axis) {
		attitude_integral[axis] =
			persistent[2 + axis] + errors[axis] * model.dt_s;
	}

	const float yaw_integral_gain = model.attitude_gain[23];

	if (policy != nullptr) {
		const float limit = policy[M::YawIntegralMomentLimit] / fmaxf(fabsf(yaw_integral_gain), 1e-9f);
		attitude_integral[2] = clamp(attitude_integral[2], -limit, limit);
	}

	float previous_roll_pitch_integral[2] {persistent[2], persistent[3]};

	if (shared_yaw) {
		for (int axis = 0; axis < 2; ++axis) {
			const float gain = model.attitude_gain[axis * M::kStateSize + 5 + axis];
			const float limit = model.roll_pitch_integral_moment_fraction * model.maximum_moment_n_m[axis]
				/ fmaxf(fabsf(gain), 1e-9f);
			previous_roll_pitch_integral[axis] = clamp(persistent[2 + axis], -limit, limit);
			// Low commanded thrust is not a landing detector. It only prevents
			// building attitude bias while very little control force is available.
			attitude_integral[axis] = base_thrust < 0.5f * model.mass_kg * model.gravity_m_s2
				? previous_roll_pitch_integral[axis]
				: clamp(previous_roll_pitch_integral[axis] + errors[axis] * model.dt_s, -limit, limit);
		}
	}

	const float state[LqiCoaxialCollectiveModel::kStateSize] {
		roll_error,
		pitch_error,
		rate_error[0],
		rate_error[1],
		rate_error[2],
		attitude_integral[0],
		attitude_integral[1],
		attitude_integral[2],
	};
	float desired_moment[3] {};
	float raw_yaw_moment = 0.0f;
	float raw_moment[3] {};

	for (int row = 0; row < 3; ++row) {
		float accumulator = 0.0f;

		for (int col = 0; col < LqiCoaxialCollectiveModel::kStateSize; ++col) {
			accumulator += model.attitude_gain[row * LqiCoaxialCollectiveModel::kStateSize + col]
				       * state[col];
		}

		raw_moment[row] = -accumulator;
		if (row == 2) { raw_yaw_moment = -accumulator; }

		desired_moment[row] =
			clamp(-accumulator,
			      -model.maximum_moment_n_m[row],
			      model.maximum_moment_n_m[row]);
	}

	const float servo_scale =
		clamp(
			desired_thrust
			/ (model.mass_kg * model.gravity_m_s2)
			* model.servo_effectiveness_scale,
			model.servo_effectiveness_scale_minimum,
			model.servo_effectiveness_scale_maximum);
	float servo_raw[3] {};

	for (int servo = 0; servo < 3; ++servo) {
		servo_raw[servo] =
			(model.servo_direction[2 * servo] * desired_moment[0]
			 + model.servo_direction[2 * servo + 1] * desired_moment[1])
			/ servo_scale;
	}

	float servo_limit = 0.0f;

	for (float value : servo_raw) {
		servo_limit = fmaxf(servo_limit, fabsf(value));
	}

	const bool servo_saturated = servo_limit > 1.0f;
	const float servo_vector_scale = servo_saturated ? 1.0f / servo_limit : 1.0f;
	float servos[3] {};

	for (int servo = 0; servo < 3; ++servo) {
		servos[servo] = clamp(servo_raw[servo] * servo_vector_scale, -1.0f, 1.0f);
	}

	float grid_common = 0.0f;
	float grid_common_min = 0.0f;
	float grid_common_max = 0.0f;
	float base_yaw_moment = 0.0f;
	float grid_yaw_moment = 0.0f;

	if (policy != nullptr) {
		// Roll/pitch take priority: only insert common deflection into the
		// remaining intersection of all three physical servo intervals.
		float common_min = -1.0f;
		float common_max = 1.0f;

		for (float value : servos) {
			common_min = fmaxf(common_min, -1.0f - value);
			common_max = fminf(common_max, 1.0f - value);
		}

		const float ramp = clamp(desired_thrust / (model.mass_kg * model.gravity_m_s2), 0.0f, 1.0f);
		const float full_common = policy[M::GridCommonCommand] * ramp;
		grid_common_min = fmaxf(common_min, fminf(full_common, 0.0f));
		grid_common_max = fminf(common_max, fmaxf(full_common, 0.0f));
		grid_common = clamp(full_common * (shared_yaw ? model.grid_common_preference_fraction : 1.0f),
			grid_common_min, grid_common_max);

		for (float &value : servos) { value = clamp(value + grid_common, -1.0f, 1.0f); }

		base_yaw_moment = model.motor_torque_coefficients[0] * upper_target_speed * upper_target_speed
			- model.motor_torque_coefficients[1] * policy[M::LowerTorqueScale] * lower_target_speed * lower_target_speed;
		grid_yaw_moment = policy[M::GridYawPerThrustPerCommand] * base_thrust * grid_common;
	}

	float motor_yaw_request = desired_moment[2] - base_yaw_moment - grid_yaw_moment;
	float motor_effectiveness[4] {};
	interpolate_vector(
		model.motor_effectiveness_points,
		model.motor_effectiveness,
		model.motor_effectiveness_points_count,
		4,
		clamp(0.5f * (upper_base + lower_base), 0.0f, 1.0f),
		motor_effectiveness);
	if (policy != nullptr) { motor_effectiveness[3] *= policy[M::LowerTorqueScale]; }

	const float combined_regularization =
		2.0f * model.motor_mix_regularization;
	const float common_thrust_slope =
		motor_effectiveness[0] + motor_effectiveness[1];
	const float differential_thrust_slope =
		motor_effectiveness[0] - motor_effectiveness[1];
	const float common_yaw_slope =
		motor_effectiveness[2] + motor_effectiveness[3];
	const float differential_yaw_slope =
		motor_effectiveness[2] - motor_effectiveness[3];
	const float determinant =
		(common_thrust_slope + combined_regularization)
		* (differential_yaw_slope + combined_regularization)
		- differential_thrust_slope * common_yaw_slope;

	if (fabsf(determinant) < 1e-12f) {
		return false;
	}

	const float common_output =
		-differential_thrust_slope * motor_yaw_request / determinant;
	const float differential_output =
		(common_thrust_slope + combined_regularization)
		* motor_yaw_request / determinant;
	const float motor_raw[2] {
		upper_base + common_output + differential_output,
		lower_base + common_output - differential_output,
	};
	float upper = clamp(motor_raw[0], 0.0f, 1.0f);
	float lower = clamp(motor_raw[1], 0.0f, 1.0f);
	bool motor_saturated =
		motor_raw[0] < 0.0f || motor_raw[0] > 1.0f
		|| motor_raw[1] < 0.0f || motor_raw[1] > 1.0f;

	bool yaw_limited = fabsf(raw_yaw_moment - desired_moment[2]) > 1e-7f;

	if (policy != nullptr) {
		// Scale the common/differential correction as a vector. This preserves
		// its local zero-thrust direction; independent endpoint clipping does not.
		const float upper_per_moment = (-differential_thrust_slope + common_thrust_slope + combined_regularization) / determinant;
		const float lower_per_moment = (-differential_thrust_slope - common_thrust_slope - combined_regularization) / determinant;
		float fraction = lqi_coaxial_collective_detail::allocate_bounded_motors(model,
			upper_base, lower_base, motor_raw[0] - upper_base, motor_raw[1] - lower_base, upper, lower);

		if (shared_yaw && fraction < 1.0f) {
			const float achieved_motor_yaw = motor_effectiveness[2] * (upper - upper_base)
				+ motor_effectiveness[3] * (lower - lower_base);
			const float grid_slope = policy[M::GridYawPerThrustPerCommand] * base_thrust;

			if (fabsf(grid_slope) > 1e-9f) {
				// Share the residual with the grid. A negative common moves toward
				// zero when it opposes positive yaw; negative yaw may use the rest
				// of the original bounded trim range. Roll/pitch remain unchanged.
				const float selected = clamp(grid_common + (motor_yaw_request - achieved_motor_yaw) / grid_slope,
					grid_common_min, grid_common_max);

				for (float &value : servos) { value = clamp(value + selected - grid_common, -1.0f, 1.0f); }

				grid_common = selected;
				grid_yaw_moment = grid_slope * grid_common;
				motor_yaw_request = desired_moment[2] - base_yaw_moment - grid_yaw_moment;
				fraction = lqi_coaxial_collective_detail::allocate_bounded_motors(model,
					upper_base, lower_base, upper_per_moment * motor_yaw_request,
					lower_per_moment * motor_yaw_request, upper, lower);
			}
		}

		motor_saturated = base_saturated || fraction < 1.0f;
		yaw_limited = yaw_limited || fraction < 1.0f;
	}

	next_persistent[0] = filtered_up_acceleration;
	next_persistent[1] =
		(policy != nullptr ? base_saturated : motor_saturated)
		? persistent[1]
		: candidate_acceleration_integral;
	const bool attitude_saturated = servo_saturated || motor_saturated;

	for (int axis = 0; axis < 3; ++axis) {
		next_persistent[2 + axis] =
			attitude_saturated
			? persistent[2 + axis]
			: attitude_integral[axis];
	}

	if (policy != nullptr) {
		// Independent axes, bounded yaw bias, and directional anti-windup:
		// permit the yaw integral to unwind even while the mixer is saturated.
		for (int axis = 0; axis < 2; ++axis) {
			if (shared_yaw) {
				const float gain = model.attitude_gain[axis * M::kStateSize + 5 + axis];
				const float achieved = desired_moment[axis] * servo_vector_scale;
				const float residual = raw_moment[axis] - achieved;
				const float change = -gain * (attitude_integral[axis] - previous_roll_pitch_integral[axis]);
				next_persistent[2 + axis] = fabsf(residual) > 1e-7f && change * residual > 0.0f
					? previous_roll_pitch_integral[axis] : attitude_integral[axis];
			} else {
				next_persistent[2 + axis] = servo_saturated ? persistent[2 + axis] : attitude_integral[axis];
			}
		}

		const float achieved_yaw = base_yaw_moment + grid_yaw_moment
			+ motor_effectiveness[2] * (upper - upper_base)
			+ motor_effectiveness[3] * (lower - lower_base);
		const float residual = raw_yaw_moment - achieved_yaw;
		const float integral_moment_change = -yaw_integral_gain * (attitude_integral[2] - persistent[4]);
		const float limit = policy[M::YawIntegralMomentLimit] / fmaxf(fabsf(yaw_integral_gain), 1e-9f);
		next_persistent[4] = clamp(yaw_limited && integral_moment_change * residual > 0.0f
			? persistent[4] : attitude_integral[2], -limit, limit);
	}

	output.upper = upper;
	output.lower = lower;
	output.servos[0] = servos[0];
	output.servos[1] = servos[1];
	output.servos[2] = servos[2];

	for (float value : output.servos) {
		if (!std::isfinite(value)) {
			return false;
		}
	}

	return std::isfinite(output.upper) && std::isfinite(output.lower);
}

inline bool LqiCoaxialCollectiveCore::step(
	const Input &input, ControlOutput &output)
{
	float next_persistent[kPersistentSize] {};

	if (!stepFromState(_model, input, _persistent, output, next_persistent)) {
		return false;
	}

	for (int index = 0; index < kPersistentSize; ++index) {
		_persistent[index] = next_persistent[index];
	}

	_last_output = output;
	return true;
}
