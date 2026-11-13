#include "PairEstimateSafety.hpp"
#include <cassert>
#include <cstdio>
#include <limits>
struct Estimate {
    enum : uint16_t { HEALTH_SOURCE_A_FRESH=4, HEALTH_SOURCE_B_FRESH=8, HEALTH_EKF_VALID=32, HEALTH_DEGRADED=1 };
    uint64_t timestamp{100000}, timestamp_sample{100000};
    float q[4]{1,0,0,0}, angular_velocity[3]{}, angular_acceleration[3]{}, specific_force[3]{0,0,-9.81f};
    uint32_t source_age_a_us{}, source_age_b_us{};
    uint8_t reset_counter{2}; uint16_t health_flags{44}; bool valid{true};
};
using namespace pair_estimate_safety;
void fault_case(const Estimate &bad, uint64_t now) {
    Guard g; Estimate good;
    assert(!g.update(false,good,100000));
    assert(g.update(true,bad,now));
    assert(g.update(true,good,100000)); // Healthy recovery cannot restart outputs.
    assert(g.update(false,good,100000)); // Automatic disarm cannot acknowledge the fault.
    g.acknowledge();
    assert(!g.update(false,good,100000)); assert(!g.update(true,good,100000));
}
int main() {
    Estimate e; assert(healthy(e,110000)); assert(!healthy(e,110001));
    fault_case(e,110001);
    e.valid=false; fault_case(e,100000); e={};
    e.timestamp=0; fault_case(e,100000); e={};
    e.timestamp_sample=0; fault_case(e,100000); e={};
    e.timestamp_sample=100001; fault_case(e,100000); e={};
    e.health_flags=4|32; fault_case(e,100000); e={};
    e.health_flags|=1; fault_case(e,100000); e={};
    e.source_age_a_us=10001; fault_case(e,100000); e={};
    e.source_age_b_us=10001; fault_case(e,100000); e={};
    e.q[0]=0; fault_case(e,100000); e={};
    e.q[1]=std::numeric_limits<float>::quiet_NaN(); fault_case(e,100000); e={};
    e.angular_velocity[0]=std::numeric_limits<float>::infinity(); fault_case(e,100000); e={};
    e.specific_force[2]=std::numeric_limits<float>::quiet_NaN(); fault_case(e,100000); e={};
    e.reset_counter=3; fault_case(e,100000); e={};
    Guard g; e.valid=false; assert(!g.update(false,e,100000));
    assert(g.update(true,e,100000)); // Preflight invalidity does not kill disarmed hardware; armed invalidity does.
    g.acknowledge(); e={}; assert(!g.update(false,e,100000));
    e.reset_counter++; assert(!g.update(false,e,100000)); assert(!g.update(true,e,100000));
    e.reset_counter++; assert(g.update(true,e,100000) && g.fault()==Fault::Reset);
    std::puts("Pair estimate kill: validity, 10 ms boundaries, source loss, numerical/reset faults and sticky recovery passed");
}
