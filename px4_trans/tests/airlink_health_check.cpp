#include "AirlinkHealth.hpp"
#include "AirlinkSession.hpp"
#include <cassert>
#include <cstdio>
#include <memory>
using namespace airlink;
int main() {
    RuntimeHealth h;
    h.run.sample(1000); h.run.sample(2000); h.run.sample(13001);
    assert(h.run.current == 11001 && h.run.maximum == 11001 && h.run.over_limit == 1);
    h.work(1000, 13000); assert(h.work_max_us == 12000);
    h.observe(1000, true); h.observe(11000, true); assert(h.stream_timeouts == 0);
    h.observe(11001, true); h.observe(12000, true); assert(h.stream_timeouts == 1);
    h.rx.sample(12001); h.observe(12001, true); h.observe(22002, true); assert(h.stream_timeouts == 2);
    h.peerImu(false); h.peerImu(false); h.peerImu(true); h.peerImu(false);
    assert(h.peer_imu_invalid_packets == 3 && h.peer_imu_invalid_events == 2);
    h.sessionChanged(); assert(!h.rx.last && h.stream_timeouts == 2 && h.run.maximum == 11001);
    h.observe(1000000, false); assert(h.stream_timeouts == 2);
    assert(health_elapsed(100, 200) == UINT32_MAX && health_elapsed(100, 0) == UINT32_MAX);
    TimingHealth t; t.session(1); assert(t.due(1000)); t.sent(1000);
    t.reply(999, 2000); assert(!t.samples); // foreign/duplicate token cannot make RTT.
    t.reply(1000, 12001); assert(t.rtt_us == 11001 && t.over_limit == 1 && t.samples == 1);
    t.reply(1000, 13000); assert(t.samples == 1);
    assert(!t.due(200999) && t.due(201000)); t.sent(201000);
    t.expire(301000); assert(!t.timeouts); t.expire(301001); assert(t.timeouts == 1);
    t.reply(201000, 301002); assert(t.samples == 1);
    t.sent(401000); t.session(2); assert(!t.pending && !t.last_reply && t.rtt_max_us == 11001);
    std::puts("Runtime scheduling, stale episodes, source invalidity and RTT checks passed");
}
