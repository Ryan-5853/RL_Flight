/****************************************************************************
 *
 *   Copyright (c) 2026 PX4 Development Team. All rights reserved.
 *
 ****************************************************************************/

#pragma once

#include "LqiControllerModel.hpp"

#include <cmath>

/**
 * Dependency-free, weight-independent nominal 4-output LQI core.
 *
 * Contract:
 * - The pilot owns the upper rotor command. The LQI owns the lower rotor and
 *   all three servos (4 outputs); lower collective base follows upper.
 * - State (13): [roll_error, pitch_error, p, q, r, upper_f, lower_f,
 *   servo_1, servo_2, servo_3, integral_roll, integral_pitch,
 *   integral_yaw_rate].
 * - Motor states use fresh Motor1/Motor2 ESC RPM when both are available and
 *   otherwise fall back to command-driven first-order observers. Servo states
 *   are command-driven because there is no servo-angle feedback.
 * - Fixed 500 Hz step (dt = 2 ms) matching the gain discretization.
 *
 * The class is host-compilable (no PX4 dependencies) so the same math is
 * exercised by the host golden-vector check and the firmware backend.
 */
class LqiControllerCore
{
public:
	struct Input {
		// Both attitudes are Hamilton [w,x,y,z] rotations from body FRD to
		// world NED. Rates are expressed in body FRD. The error convention is
		// target-to-current, matching flight_controller.math.
		float attitude_q_wb[4];
		float target_attitude_q_wb[4];
		float angular_velocity_b[3];
		float target_angular_velocity_b[3];
		// Bias-corrected acceleration including gravity in body FRD.
		// Existing LQI variants ignore this field.
		float acceleration_b[3];
		float collective_base; // normalized pilot command [0, 1]
		float motor_rpm[2];
		bool motor_rpm_valid;
	};

	struct ControlOutput {
		float upper;   // external upper rotor command [0, 1]
		float lower;   // LQI lower rotor command [0, 1]
		float servos[3]; // LQI servo commands [-1, 1]
	};

	static constexpr int kPersistentSize = LqiControllerModel::kPersistentSize;

	// Safety limits applied inside the LQI step so verification, saturation
	// handling and the firmware backend all observe the same contract.
	static constexpr float kIdleCollectiveThreshold{0.0f};
	static constexpr float kMaxUpperLowerDelta{0.2f};

	explicit LqiControllerCore(const LqiControllerModel &model) : _model(model) {}

	void reset();

	/**
	 * One 500 Hz step from the persistent state held by this instance.
	 * Returns false when the computed output is non-finite.
	 */
	bool step(const Input &input, ControlOutput &output);

	/**
	 * Pure single-step evaluation used by the verification protocol and the
	 * host golden-vector check. `persistent` is [actuator(5), integral(3)].
	 */
	static bool stepFromState(const LqiControllerModel &model,
				  const Input &input,
				  const float persistent[kPersistentSize],
				  ControlOutput &output,
				  float next_persistent[kPersistentSize]);

	const float *persistent() const { return _persistent; }
	const ControlOutput &lastOutput() const { return _last_output; }

private:
	const LqiControllerModel &_model;
	float _persistent[kPersistentSize]{};
	ControlOutput _last_output{};
};

inline void LqiControllerCore::reset()
{
	for (float &value : _persistent) {
		value = 0.0f;
	}

	_last_output = {};
}

inline bool LqiControllerCore::stepFromState(
	const LqiControllerModel &model,
	const Input &input,
	const float persistent[kPersistentSize],
	ControlOutput &output,
	float next_persistent[kPersistentSize])
{
	if (model.schema_version != LqiControllerModel::kSchemaVersion
	    || model.world_frame != LqiControllerModel::WorldFrame::NED
	    || model.body_frame != LqiControllerModel::BodyFrame::FRD
	    || model.quaternion_convention
	       != LqiControllerModel::QuaternionConvention::HamiltonWxyzBodyToWorld
	    || model.collective_contract
	       != LqiControllerModel::CollectiveContract::ExternalUpperLqiLowerAndServos
	    || model.actuator_observer != LqiControllerModel::ActuatorObserver::MotorRpmOrCommandDriven
	    || model.gain == nullptr || model.decay == nullptr || model.slope_eff == nullptr
	    || model.integral_limit == nullptr || model.motor_command_to_rad_s == nullptr
	    || model.motor_table_points < 2 || !(model.dt_s > 0.0f)) {
		return false;
	}

	// q_error = conjugate(q_target_wb) * q_current_wb. This is the
	// target-to-current local rotation used by the Python nominal controller.
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

	// Yaw tracks rate only (the yaw component of the attitude error is not a
	// state), while all three body rates use current-minus-target error.
	const float p_error = input.angular_velocity_b[0] - input.target_angular_velocity_b[0];
	const float q_error = input.angular_velocity_b[1] - input.target_angular_velocity_b[1];
	const float r_error = input.angular_velocity_b[2] - input.target_angular_velocity_b[2];

	const float collective = (input.collective_base < 0.0f) ? 0.0f
				 : ((input.collective_base > 1.0f) ? 1.0f : input.collective_base);

	// No throttle command: the controller stays idle, produces no actuator
	// output and leaves its persistent state untouched. This covers both the
	// pre-takeoff state (fresh instance, zero state) and any brief throttle
	// release after arming.
	if (collective <= kIdleCollectiveThreshold) {
		output.upper = 0.0f;
		output.lower = 0.0f;
		output.servos[0] = 0.0f;
		output.servos[1] = 0.0f;
		output.servos[2] = 0.0f;

		for (int index = 0; index < kPersistentSize; ++index) {
			next_persistent[index] = persistent[index];
		}

		return true;
	}

	// Match Controller's piecewise-linear motor target lookup. Table entries
	// are [normalized command, rad/s] pairs shared by both coaxial motors.
	int upper_point = 1;

	while (upper_point < model.motor_table_points - 1
	       && collective > model.motor_command_to_rad_s[2 * upper_point]) {
		++upper_point;
	}

	const int lower_point = upper_point - 1;
	const float command_0 = model.motor_command_to_rad_s[2 * lower_point];
	const float speed_0 = model.motor_command_to_rad_s[2 * lower_point + 1];
	const float command_1 = model.motor_command_to_rad_s[2 * upper_point];
	const float speed_1 = model.motor_command_to_rad_s[2 * upper_point + 1];
	const float command_clamped = (collective < model.motor_command_to_rad_s[0])
				      ? model.motor_command_to_rad_s[0]
				      : ((collective > model.motor_command_to_rad_s[2 * (model.motor_table_points - 1)])
					 ? model.motor_command_to_rad_s[2 * (model.motor_table_points - 1)] : collective);
	const float target_speed = speed_0 + (command_clamped - command_0) * (speed_1 - speed_0)
				   / (command_1 - command_0);
	const float target_upper_dev = target_speed - model.upper_trim_rad_s;
	const float target_lower_dev = target_speed - model.lower_trim_rad_s;
	const float upper_motor_error = input.motor_rpm_valid
		? std::fabs(input.motor_rpm[0]) * model.rpm_to_rad_s - model.upper_trim_rad_s - target_upper_dev
		: persistent[0] - target_upper_dev;
	const float lower_motor_error = input.motor_rpm_valid
		? std::fabs(input.motor_rpm[1]) * model.rpm_to_rad_s - model.lower_trim_rad_s - target_lower_dev
		: persistent[1] - target_lower_dev;

	const float x[LqiControllerModel::kStateSize] {
		roll_error,
		pitch_error,
		p_error,
		q_error,
		r_error,
		upper_motor_error,
		lower_motor_error,
		persistent[2],
		persistent[3],
		persistent[4],
		persistent[5],
		persistent[6],
		persistent[7],
	};

	float delta[LqiControllerModel::kOutputSize] {};

	for (int row = 0; row < LqiControllerModel::kOutputSize; ++row) {
		float accumulator = 0.0f;

		for (int col = 0; col < LqiControllerModel::kStateSize; ++col) {
			accumulator += model.gain[row * LqiControllerModel::kStateSize + col] * x[col];
		}

		delta[row] = -accumulator;
	}

	const float lower_raw = collective + delta[0];
	const float servo_raw[3] { delta[1], delta[2], delta[3] };
	const float upper = collective;
	// Keep the lower-rotor opening within 20 percentage points of the pilot
	// upper-rotor command. The bounds are additionally constrained to [0, 1].
	const float lower_min = (upper - kMaxUpperLowerDelta < 0.0f)
				? 0.0f : (upper - kMaxUpperLowerDelta);
	const float lower_max = (upper + kMaxUpperLowerDelta > 1.0f)
				? 1.0f : (upper + kMaxUpperLowerDelta);
	const float lower = (lower_raw < lower_min) ? lower_min
			    : ((lower_raw > lower_max) ? lower_max : lower_raw);

	float servos[3] {};

	for (int index = 0; index < 3; ++index) {
		servos[index] = (servo_raw[index] < -1.0f) ? -1.0f
				: ((servo_raw[index] > 1.0f) ? 1.0f : servo_raw[index]);
	}

	const bool saturated = (lower_raw < lower_min)
			       || (lower_raw > lower_max)
			       || (servo_raw[0] < -1.0f) || (servo_raw[0] > 1.0f)
			       || (servo_raw[1] < -1.0f) || (servo_raw[1] > 1.0f)
			       || (servo_raw[2] < -1.0f) || (servo_raw[2] > 1.0f);

	// Integral states: [roll_error, pitch_error, yaw_rate_error], frozen
	// on command saturation to prevent integral windup.
	const float errors[3] { roll_error, pitch_error, r_error };
	float integral_next[3] {};

	for (int axis = 0; axis < 3; ++axis) {
		const float candidate = persistent[5 + axis] + errors[axis] * model.dt_s;
		const float clamped = (candidate < -model.integral_limit[axis]) ? -model.integral_limit[axis]
				      : ((candidate > model.integral_limit[axis]) ? model.integral_limit[axis] : candidate);
		integral_next[axis] = saturated ? persistent[5 + axis] : clamped;
	}

	// Command-driven actuator observers: state += decay*state + slope_eff*u.
	const float command_deviation[5] {
		upper - model.upper_trim,
		lower - model.lower_trim,
		servos[0],
		servos[1],
		servos[2],
	};

	for (int actuator = 0; actuator < 5; ++actuator) {
		next_persistent[actuator] =
			model.decay[actuator] * persistent[actuator]
			+ model.slope_eff[actuator] * command_deviation[actuator];
	}

	if (input.motor_rpm_valid) {
		next_persistent[0] = std::fabs(input.motor_rpm[0]) * model.rpm_to_rad_s - model.upper_trim_rad_s;
		next_persistent[1] = std::fabs(input.motor_rpm[1]) * model.rpm_to_rad_s - model.lower_trim_rad_s;
	}

	for (int axis = 0; axis < 3; ++axis) {
		next_persistent[5 + axis] = integral_next[axis];
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

inline bool LqiControllerCore::step(const Input &input, ControlOutput &output)
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
