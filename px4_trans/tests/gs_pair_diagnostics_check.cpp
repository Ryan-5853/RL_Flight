#include "../px4/src/modules/gs_link/GsLinkDiagnostics.hpp"
#include "../px4/src/modules/nn_control/PairGsGate.hpp"
#include <cassert>
#include <cstdio>
#include <cmath>
#include <limits>

template<typename T> void emit(unsigned id, const T& value, FILE* output)
{
    std::fprintf(output, "%u ", id);
    const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
    for (unsigned i = 0; i < sizeof(T); ++i) std::fprintf(output, "%02x", bytes[i]);
    std::fprintf(output, "\n");
}
int main(int argc, char** argv)
{
    using namespace gs_link;
    NodeDiagnostics node{}; node.role = 1; node.nav_state = 25;
    node.timestamp_us = 0x0102030405060708ULL; node.airlink_timestamp_us = 123456;
    node.flags = 8192 | 1024 | 256 | 128 | 8 | 4 | 1; node.session_id = 0x11223344;
    node.peer_session_id = 0x55667788; node.airlink_state = 8; node.airlink_version = 2;
    node.health_flags = 127; node.run_interval_max_us = 12501; node.rtt_max_us = 15001; node.rtt_us = 2500; node.rtt_samples = 5;
    node.rx_age_us = UINT32_MAX; node.stream_timeouts = 2; node.local_core_stale_events = 3;
    node.logger_flags = 3; node.logger_total_kb = 123.5f;
    ObserverDiagnostics observer{}; observer.role = 1; observer.timestamp_us = 123400;
    observer.q[0] = 1.f; observer.weights[0] = observer.weights[1] = .5f;
    observer.flags = 63; observer.angular_velocity[2] = -.75f;
    ControllerDiagnostics controller{}; controller.role = 1; controller.enabled = 15;
    controller.timestamp_us = 123400; controller.requested_wrench[0] = 12.9f;
    controller.residual_wrench[2] = -.0125f; controller.reference_collective = .5f;
    controller.attitude_error[1] = .25f;
    controller.moment_attitude[1] = -.04f;
    controller.filtered_vertical_acceleration = .2f;
    controller.installation_yaw[0] = -.523598776f;
    controller.installation_yaw[1] = .523598776f;
    controller.arm_m = .075f; controller.loop_dt_s = .002f; controller.trace_flags = 1;
    FlightDiagnostics flight{}; flight.role = 2; flight.timestamp_us = 123456;
    flight.motors[0] = .2f; flight.motors[1] = .3f;
    flight.servos[2] = -.4f; flight.rc[3] = -1.f;
    flight.q[0] = 1.f; flight.voltage_v = 16.8f;
    flight.received_motors[0] = std::numeric_limits<float>::quiet_NaN();
    assert(sizeof(node) + kHeaderSize + kCrcSize < 512);
    assert(sizeof(observer) + kHeaderSize + kCrcSize < 512);
    assert(sizeof(controller) + kHeaderSize + kCrcSize < 512);
    assert(sizeof(flight) + kHeaderSize + kCrcSize < 512);
    PairGsGate gate;
    assert(!gate.update(false, true, true));
    assert(!gate.update(true, true, true));
    assert(!gate.update(true, true, false));
    assert(gate.update(true, true, true));
    assert(!gate.update(true, false, true));
    assert(!gate.update(true, true, true));
    assert(!gate.update(true, true, false));
    assert(gate.update(true, true, true));
    assert(!gate.update(false, true, true));
    if (argc > 1) {
        FILE* out = std::fopen(argv[1], "w"); assert(out);
        emit(kNodeDiagnosticsId, node, out); emit(kObserverDiagnosticsId, observer, out);
        emit(kControllerDiagnosticsId, controller, out); emit(kFlightDiagnosticsId, flight, out);
        std::fclose(out);
    }
    std::puts("GS-Link pair diagnostics ABI and control gate checks passed");
}
