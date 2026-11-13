#include "MaintainedArmSwitch.hpp"
#include "AirlinkSession.hpp"
#include <cassert>
#include <cstdio>

int main()
{
	using airlink::MaintainedArmSwitch;
	MaintainedArmSwitch sw;
	// Boot with ON: no implicit arm, even with repeated fresh ON samples.
	auto result = sw.update(100, true, true, false);
	assert(!result.request_arm && !result.arm_allowed);
	result = sw.update(101, true, true, false);
	assert(!result.request_arm && !result.arm_allowed);
	// OFF explicitly acknowledges disarm even when the endpoint never armed.
	result = sw.update(102, true, false, true);
	assert(result.request_disarm && result.off && !result.arm_allowed);
	for (unsigned i = 103; i < 110; ++i) {
		result = sw.update(i, true, false, true);
		assert(!result.request_disarm && result.off);
	}
	result = sw.update(110, true, true, false);
	assert(result.request_arm && result.arm_allowed && !result.off);
	for (unsigned i = 111; i < 120; ++i) {
		result = sw.update(i, true, true, false);
		assert(!result.request_arm && result.arm_allowed);
	}
	// RC loss + reconnect while ON cannot retrigger a failed arm transaction.
	result = sw.update(120, false, true, false);
	assert(!result.arm_allowed && !result.request_disarm);
	result = sw.update(121, true, true, false);
	assert(!result.request_arm && !result.arm_allowed);
	result = sw.update(122, true, false, true);
	assert(result.request_disarm);
	// Duplicate or delayed samples cannot replay an ON edge.
	result = sw.update(122, true, true, false);
	assert(!result.request_arm && !result.arm_allowed && result.off);
	result = sw.update(123, true, true, false);
	assert(result.request_arm && result.arm_allowed);
	result = sw.update(124, true, false, false); // unmapped/middle
	assert(!result.arm_allowed);
	result = sw.update(125, true, true, false);
	assert(!result.request_arm && !result.arm_allowed);
	// Holding OFF across loss must not acknowledge a new fault automatically.
	MaintainedArmSwitch off;
	assert(off.update(100, true, false, true).request_disarm);
	off.update(101, false, false, true);
	assert(!off.update(102, true, false, true).request_disarm);

	// A mapped OFF edge cancels an in-flight slave arm transaction and clears
	// its lock only after actual local disarm; ON held is never the recovery.
	MaintainedArmSwitch recovery;
	recovery.update(1, true, false, true);
	assert(recovery.update(2, true, true, false).request_arm);
	airlink::Session session(1);
	airlink::LocalStatus local{};
	local.local_rc_valid = local.local_ready = local.safety_gate_allow = local.pair_mode_selected = true;
	local.armed = true;
	local.pair_reattach_latched = true;
	session.start(airlink::Role::Master, 10, local, 1000);
	assert(session.pair_reattach_latched());
	assert(!recovery.update(3, true, true, false).request_disarm);
	assert(recovery.update(4, true, false, true).request_disarm);
	assert(!session.reset_after_operator_disarm(1001)); // Must really disarm first.
	session.set_armed(false);
	assert(session.reset_after_operator_disarm(1002));
	assert(!session.pair_reattach_latched() && session.session_id() == 11);
	assert(!recovery.update(5, true, false, true).request_disarm);
	assert(recovery.update(6, true, true, false).request_arm);
	// Slave RC ON may arrive after the master's PAIR_ARM. Defer within the
	// transaction, issue once after the local edge, cancel on OFF or timeout.
	MaintainedArmSwitch delayed;
	using Pending = MaintainedArmSwitch::PendingArm;
	delayed.update(1, true, false, true);
	delayed.deferArm(100);
	assert(delayed.pollArm(10, true, false) == Pending::None);
	delayed.update(20, true, true, false);
	assert(delayed.pollArm(21, true, false) == Pending::Issue);
	assert(delayed.pollArm(22, true, false) == Pending::None);
	delayed.deferArm(200);
	delayed.update(30, true, false, true);
	assert(delayed.pollArm(31, true, true) == Pending::Reject);
	delayed.deferArm(40);
	assert(delayed.pollArm(40, true, false) == Pending::Reject);
	delayed.deferArm(100);
	assert(delayed.pollArm(50, false, false) == Pending::Reject);
	std::puts("maintained ARM switch edges, loss/reconnect and fault recovery checks passed");
}
