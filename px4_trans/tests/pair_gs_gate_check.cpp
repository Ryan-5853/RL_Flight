#include "../px4/src/modules/nn_control/PairGsGate.hpp"
#include <cassert>
#include <cstdio>
int main()
{
	PairGsGate gate;
	assert(!gate.update(false, true, true)); // request before armed is ignored
	assert(!gate.update(true, true, true));  // no automatic takeover on arm
	assert(!gate.update(true, true, false));
	assert(gate.update(true, true, true));
	assert(!gate.update(true, false, true)); // timeout returns to RC
	assert(!gate.update(true, true, true));  // stale request cannot resume
	assert(!gate.update(true, true, false));
	assert(gate.update(true, true, true));
	assert(!gate.update(false, true, true)); // mode/role/disarm gate wins
	assert(!gate.update(true, true, true));
	std::puts("Pair GS gate checks passed");
}
