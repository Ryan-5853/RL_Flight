/****************************************************************************
 *
 *   Copyright (c) 2026 PX4 Development Team. All rights reserved.
 *
 ****************************************************************************/

#include "../px4/src/modules/nn_control/GroundStationGate.hpp"

#include <cstdio>

namespace
{

struct Harness {
	GroundStationGate gate{};
	GroundStationGate::Config config{};
	uint64_t now{0};

	GroundStationGate::Source update(bool armed, bool rc_valid, bool mode_slot_valid,
					 uint8_t mode_slot, bool ground_station_valid,
					 bool external_mode_active = true)
	{
		GroundStationGate::Input input{};
		input.now = now;
		input.armed = armed;
		input.rc_valid = rc_valid;
		input.mode_slot_valid = mode_slot_valid;
		input.mode_slot = mode_slot;
		input.external_mode_active = external_mode_active;
		input.ground_station_valid = ground_station_valid;
		return gate.update(config, input);
	}
};

int check(bool condition, const char *name)
{
	if (!condition) {
		std::printf("FAILED: %s\n", name);
		return 1;
	}

	return 0;
}

} // namespace

int main()
{
	int failures = 0;
	constexpr uint64_t ms = 1000;

	{
		Harness h{};
		h.config.enabled = true;
		h.config.mode_slot = 4;
		h.config.entry_delay_us = 100 * ms;

		failures += check(h.update(false, true, true, 1, true)
				  == GroundStationGate::Source::Rc,
				  "mode 4 while disarmed stays on RC");
		failures += check(!h.gate.mode_selected(),
				  "mode selection is not active while disarmed");

		h.now = 1 * ms;
		failures += check(h.update(true, true, true, 1, true)
				  == GroundStationGate::Source::Rc,
				  "arming in a non-ground-station slot stays on RC");

		h.now = 2 * ms;
		failures += check(h.update(true, true, true, 4, true)
				  == GroundStationGate::Source::Rc,
				  "mode 4 waits for entry delay");
		h.now = 2 * ms + 99 * ms;
		failures += check(h.update(true, true, true, 4, true)
				  == GroundStationGate::Source::Rc,
				  "mode 4 still RC at 99 ms");
		h.now = 2 * ms + 100 * ms;
		failures += check(h.update(true, true, true, 4, true)
				  == GroundStationGate::Source::GroundStation,
				  "mode 4 enters ground-station mode at 100 ms");

		h.now = 3 * ms;
		failures += check(h.update(true, true, true, 5, true)
				  == GroundStationGate::Source::Rc,
				  "leaving mode 4 immediately interrupts ground station");
		failures += check(h.update(true, true, true, 4, true)
				  == GroundStationGate::Source::Rc,
				  "re-entering mode 4 after delay restarts entry");

		h.now = 4 * ms;
		failures += check(h.update(false, true, true, 4, true)
				  == GroundStationGate::Source::Rc,
				  "disarm immediately leaves ground-station mode");

		h.now = 5 * ms;
		failures += check(h.update(true, true, true, 4, true)
				  == GroundStationGate::Source::Rc,
				  "arming while already in mode 4 stays on RC");
	}

	{
		Harness h{};
		h.config.enabled = true;
		h.config.mode_slot = 4;
		h.config.entry_delay_us = 100 * ms;

		failures += check(h.update(true, true, true, 1, true)
				  == GroundStationGate::Source::Rc,
				  "arm in manual");
		h.now = 1 * ms;
		failures += check(h.update(true, true, true, 4, true)
				  == GroundStationGate::Source::Rc,
				  "start ground-station entry");
		h.now = 1 * ms + 100 * ms;
		failures += check(h.update(true, true, true, 4, true)
				  == GroundStationGate::Source::GroundStation,
				  "ground station active");

		h.now = 2 * ms;
		failures += check(h.update(true, true, true, 4, false)
				  == GroundStationGate::Source::Rc,
				  "ground-station timeout falls back to RC immediately");

		h.now = 3 * ms;
		failures += check(h.update(true, true, true, 4, true)
				  == GroundStationGate::Source::Rc,
				  "reconnected ground station does not auto-resume");
		h.now = 4 * ms;
		failures += check(h.update(true, true, true, 1, true)
				  == GroundStationGate::Source::Rc,
				  "leave mode 4 before re-entry");
		h.now = 5 * ms;
		failures += check(h.update(true, true, true, 4, true)
				  == GroundStationGate::Source::Rc,
				  "reselect mode 4");
		h.now = 5 * ms + 100 * ms;
		failures += check(h.update(true, true, true, 4, true)
				  == GroundStationGate::Source::GroundStation,
				  "explicit mode selection re-enters");

		h.now = 6 * ms;
		failures += check(h.update(true, false, true, 4, true)
				  == GroundStationGate::Source::None,
				  "RC loss removes the ground-station source");
	}

	{
		Harness h{};
		h.config.enabled = true;
		h.config.mode_slot = 4;
		h.config.entry_delay_us = 100 * ms;

		failures += check(h.update(true, true, true, 1, true)
				  == GroundStationGate::Source::Rc,
				  "slot 1: arm transition");
		h.now = 1 * ms;
		failures += check(h.update(true, true, true, 4, true)
				  == GroundStationGate::Source::Rc,
				  "slot 6: waits for entry delay");
		h.now = 101 * ms;
		failures += check(h.update(true, true, true, 4, true)
				  == GroundStationGate::Source::GroundStation,
				  "slot 6: ground-station mode entered");
	}

	{
		Harness h{};
		h.config.enabled = false;
		failures += check(h.update(true, true, true, 1, true)
				  == GroundStationGate::Source::Rc,
				  "disabled ground-station path stays on RC");
		failures += check(h.update(true, true, true, 4, true)
				  == GroundStationGate::Source::Rc,
				  "disabled ground-station path never enters");
	}

	{
		Harness h{};
		h.config.enabled = true;
		h.config.mode_slot = 4;
		h.config.entry_delay_us = 100 * ms;
		failures += check(h.update(true, true, true, 1, true)
				  == GroundStationGate::Source::Rc,
				  "invalid switch: arm transition");
		h.now = 1 * ms;
		failures += check(h.update(true, true, true, 1, true)
				  == GroundStationGate::Source::Rc,
				  "invalid switch: manual fallback");
		h.now = 2 * ms;
		failures += check(h.update(true, true, false, 4, true)
				  == GroundStationGate::Source::Rc,
				  "invalid switch sample stays on RC");
	}

	{
		Harness h{};
		h.config.enabled = true;
		h.config.mode_slot = 0;
		failures += check(h.update(true, true, true, 1, true)
				  == GroundStationGate::Source::Rc,
				  "invalid target mode slot stays on RC");
		h.now = 1 * ms;
		failures += check(h.update(true, true, true, 4, true)
				  == GroundStationGate::Source::Rc,
				  "invalid target mode slot never enters");
	}

	{
		Harness h{};
		h.config.enabled = true;
		h.config.mode_slot = 4;
		h.config.entry_delay_us = 100 * ms;

		failures += check(h.update(true, true, true, 1, true)
				  == GroundStationGate::Source::Rc,
				  "async mode: arm in manual");
		h.now = 1 * ms;
		failures += check(h.update(true, true, true, 4, true, false)
				  == GroundStationGate::Source::Rc,
				  "async mode: slot 6 waits for external nav_state");
		h.now = 2 * ms;
		failures += check(h.update(true, true, true, 4, true, true)
				  == GroundStationGate::Source::Rc,
				  "async mode: external nav_state starts entry delay");
		h.now = 2 * ms + 99 * ms;
		failures += check(h.update(true, true, true, 4, true, true)
				  == GroundStationGate::Source::Rc,
				  "async mode: still RC at 99 ms");
		h.now = 2 * ms + 100 * ms;
		failures += check(h.update(true, true, true, 4, true, true)
				  == GroundStationGate::Source::GroundStation,
				  "async mode: slot 6 becomes active without reselection");

		h.now = 3 * ms;
		failures += check(h.update(true, true, true, 4, true, false)
				  == GroundStationGate::Source::Rc,
				  "async mode: losing external nav_state interrupts control");
		h.now = 4 * ms;
		failures += check(h.update(true, true, true, 4, true, true)
				  == GroundStationGate::Source::Rc,
				  "async mode: nav_state recovery does not auto-resume");
		h.now = 5 * ms;
		failures += check(h.update(true, true, true, 1, true, false)
				  == GroundStationGate::Source::Rc,
				  "async mode: leave slot 6 after interruption");
		h.now = 6 * ms;
		failures += check(h.update(true, true, true, 4, true, true)
				  == GroundStationGate::Source::Rc,
				  "async mode: explicit reselect restarts entry");
		h.now = 6 * ms + 100 * ms;
		failures += check(h.update(true, true, true, 4, true, true)
				  == GroundStationGate::Source::GroundStation,
				  "async mode: explicit reselect re-enters");
	}

	if (failures != 0) {
		std::printf("ground-station gate: %d failure(s)\n", failures);
		return 1;
	}

	std::printf("ground-station gate: all checks passed\n");
	return 0;
}
