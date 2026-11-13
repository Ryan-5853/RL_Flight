#include "ActuatorOutputValidation.hpp"
#include "../px4/src/modules/airlink/AirlinkProtocol.hpp"
#include <cstdio>
#include <limits>
#include <initializer_list>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)

int main()
{
	ActuatorSetpoint output{{.12f, .056f}, {-.49999997f, 1.00000012f, -.49999997f}};
	CHECK(actuator_output_validation::validateAndLimit(output));
	CHECK(output.motors[0] == .12f && output.motors[1] == .056f);
	CHECK(output.servos[1] == 1.f);
	output.servos[0] = -1.00000012f;
	output.servos[1] = 2.f;
	CHECK(actuator_output_validation::validateAndLimit(output));
	CHECK(output.servos[0] == -1.f && output.servos[1] == 1.f);
	CHECK(output.motors[0] == .12f && output.motors[1] == .056f);
	for (float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
		output = {{.12f, .056f}, {0.f, bad, 0.f}};
		CHECK(!actuator_output_validation::validateAndLimit(output));
	}
	for (float bad : {-0.000001f, 1.00000012f, std::numeric_limits<float>::quiet_NaN()}) {
		output = {{bad, .056f}, {0.f, 0.f, 0.f}};
		CHECK(!actuator_output_validation::validateAndLimit(output));
	}
	// Exercise every quantized servo value through the real wire codec.
	for (int code = 0; code <= 1000; ++code) {
		airlink::ActuatorCommand cmd{};
		cmd.upper_motor = .12f;
		cmd.lower_motor = .056f;
		for (float &v : cmd.servos) v = (code - 500) / 500.f;
		uint8_t payload[airlink::kActuatorPayloadSize]{};
		CHECK(airlink::encode_actuator_command(cmd, payload));
		airlink::ActuatorCommand decoded{};
		CHECK(airlink::decode_actuator_command(payload, sizeof(payload), decoded));
		for (float v : decoded.servos) {
			CHECK(v >= -1.f && v <= 1.f);
			if (code == 0) CHECK(v == -1.f);
			if (code == 500) CHECK(v == 0.f);
			if (code == 1000) CHECK(v == 1.f);
		}
		output = {{decoded.upper_motor, decoded.lower_motor}, {decoded.servos[0], decoded.servos[1], decoded.servos[2]}};
		CHECK(actuator_output_validation::validateAndLimit(output));
		CHECK(output.motors[0] == decoded.upper_motor && output.motors[1] == decoded.lower_motor);
	}
	std::puts("actuator output validation checks passed");
}
