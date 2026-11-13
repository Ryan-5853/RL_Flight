/****************************************************************************
 *
 *   Copyright (c) 2026 PX4 Development Team. All rights reserved.
 *
 ****************************************************************************/

#pragma once

#include "LqiCoaxialCollectiveModel.hpp"
#include "LqiCompositeControllerModel.hpp"
#include "LqiControllerCore.hpp"

#include <cmath>

/**
 * PX4 manual-control to LQI reference adapter.
 *
 * PX4 and the simulator both provide Hamilton q_wb ([w,x,y,z], body FRD to
 * world NED). Roll/pitch sticks command heading-relative Euler angles; yaw is
 * rate mode. PX4 positive pitch stick means nose down, so target pitch has the
 * opposite sign of the raw stick value.  The adapter only consumes the manual
 * reference constants of the model, so the same implementation serves both
 * the 13-state nominal and the 19-state identified descriptors.
 */
template <typename Model>
class LqiManualReferenceT
{
public:
	explicit LqiManualReferenceT(const Model &model) : _model(model) {}

	void reset()
	{
		_initialized = false;
		_target_yaw = 0.0f;

		for (float &value : _filtered_stick) {
			value = 0.0f;
		}
	}

	bool build(const float current_q_wb[4],
		   const float angular_velocity_b[3],
		   float rc_roll, float rc_pitch, float rc_yaw, float rc_throttle,
		   LqiControllerCore::Input &output)
	{
		for (int index = 0; index < 4; ++index) {
			output.attitude_q_wb[index] = current_q_wb[index];
		}

		for (int index = 0; index < 3; ++index) {
			output.angular_velocity_b[index] = angular_velocity_b[index];
			output.target_angular_velocity_b[index] = 0.0f;
		}

		const float raw_stick[3] {clampUnit(rc_roll), clampUnit(rc_pitch), clampUnit(rc_yaw)};

		for (int axis = 0; axis < 3; ++axis) {
			_filtered_stick[axis] = _model.manual_command_decay[axis] * _filtered_stick[axis]
						+ (1.0f - _model.manual_command_decay[axis]) * raw_stick[axis];
		}

		const float roll = _filtered_stick[0] * _model.max_roll_angle_rad;
		const float pitch = -_filtered_stick[1] * _model.max_pitch_angle_rad;
		output.target_angular_velocity_b[2] = _filtered_stick[2] * _model.max_yaw_rate_rad_s;
		output.collective_base = (clampUnit(rc_throttle) + 1.0f) * 0.5f;

		// Initialize target yaw from measured heading, then integrate the filtered
		// yaw-rate command exactly like VirtualPilotCommandSource. Yaw angle itself
		// is removed from the LQI state after forming the relative quaternion.
		const float w = current_q_wb[0];
		const float x = current_q_wb[1];
		const float y = current_q_wb[2];
		const float z = current_q_wb[3];
		const float heading = std::atan2(2.0f * (w * z + x * y),
						 1.0f - 2.0f * (y * y + z * z));

		if (!_initialized) {
			_target_yaw = heading;
			_initialized = true;
		}

		_target_yaw += output.target_angular_velocity_b[2] * _model.dt_s;
		constexpr float kPi{3.14159265358979323846f};

		if (_target_yaw > kPi) {
			_target_yaw -= 2.0f * kPi;
		} else if (_target_yaw < -kPi) {
			_target_yaw += 2.0f * kPi;
		}

		const float cr = std::cos(0.5f * roll);
		const float sr = std::sin(0.5f * roll);
		const float cp = std::cos(0.5f * pitch);
		const float sp = std::sin(0.5f * pitch);
		const float cy = std::cos(0.5f * _target_yaw);
		const float sy = std::sin(0.5f * _target_yaw);
		output.target_attitude_q_wb[0] = cr * cp * cy + sr * sp * sy;
		output.target_attitude_q_wb[1] = sr * cp * cy - cr * sp * sy;
		output.target_attitude_q_wb[2] = cr * sp * cy + sr * cp * sy;
		output.target_attitude_q_wb[3] = cr * cp * sy - sr * sp * cy;

		for (float value : output.attitude_q_wb) {
			if (!std::isfinite(value)) {
				return false;
			}
		}

		for (float value : output.angular_velocity_b) {
			if (!std::isfinite(value)) {
				return false;
			}
		}

		return std::isfinite(roll) && std::isfinite(pitch) && std::isfinite(_target_yaw)
		       && std::isfinite(output.target_angular_velocity_b[2])
		       && std::isfinite(output.collective_base);
	}

private:
	static float clampUnit(float value)
	{
		return (value < -1.0f) ? -1.0f : ((value > 1.0f) ? 1.0f : value);
	}

	const Model &_model;
	float _filtered_stick[3] {};
	float _target_yaw{0.0f};
	bool _initialized{false};
};

using LqiManualReference = LqiManualReferenceT<LqiControllerModel>;
using LqiCompositeManualReference = LqiManualReferenceT<LqiCompositeControllerModel>;
using LqiCoaxialManualReference = LqiManualReferenceT<LqiCoaxialCollectiveModel>;
