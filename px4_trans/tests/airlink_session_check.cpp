#include "AirlinkSession.hpp"

#include <cstdio>
#include <cstdlib>
#include <memory>

namespace
{

int g_failures = 0;

#define CHECK(condition) do { \
	if (!(condition)) { \
		std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
		++g_failures; \
	} \
} while (0)

airlink::LocalStatus ready_local(bool master)
{
	airlink::LocalStatus local{};
	local.local_ready = true;
	local.local_rc_valid = true;
	local.safety_gate_allow = true;
	local.pair_mode_selected = true;
	(void)master;
	return local;
}

void pump(airlink::Session &from, airlink::Session &to, uint64_t now_us)
{
	airlink::Frame frame{};

	while (from.pop_tx(frame)) {
		to.receive(frame, now_us);
	}
}

void exchange_heartbeat(airlink::Session &master, airlink::Session &slave, uint64_t now_us)
{
	(void)master.tick(now_us);
	(void)slave.tick(now_us);
	pump(master, slave, now_us);
	pump(slave, master, now_us);
}

void establish_ready(airlink::Session &master, airlink::Session &slave, uint64_t start_us)
{
	exchange_heartbeat(master, slave, start_us);
	pump(master, slave, start_us + 1);
	pump(slave, master, start_us + 2);
	pump(master, slave, start_us + 3);
	pump(slave, master, start_us + 4);

	CHECK(master.state() == airlink::LinkState::Ready);
	CHECK(slave.state() == airlink::LinkState::Ready);
	exchange_heartbeat(master, slave, start_us + 2'000);
}

void establish_pair_active(airlink::Session &master, airlink::Session &slave,
			   uint64_t start_us)
{
	establish_ready(master, slave, start_us);
	CHECK(master.request_control_enable(start_us + 20'000));
	pump(master, slave, start_us + 20'001);
	pump(slave, master, start_us + 20'002);
	CHECK(master.state() == airlink::LinkState::PairControlReady);
	CHECK(slave.state() == airlink::LinkState::PairControlReady);
	CHECK(master.request_pair_arm(start_us + 30'000));
	airlink::Frame arm_duplicate{};
	CHECK(master.pop_tx(arm_duplicate));
	slave.receive(arm_duplicate, start_us + 30'001);
	CHECK(slave.state() == airlink::LinkState::PairArmPending);
	CHECK(slave.notify_local_arm_result(true, start_us + 30'002));
	pump(slave, master, start_us + 30'003);
	CHECK(master.state() == airlink::LinkState::PairArmPending);
	CHECK(master.notify_local_arm_result(true, start_us + 30'004));
	CHECK(master.state() == airlink::LinkState::PairArmedGround);

	airlink::ActuatorCommand command{};
	command.upper_motor = command.lower_motor = 0.f;
	command.servos[0] = command.servos[1] = command.servos[2] = 0.f;
	CHECK(master.queue_actuator_command(command, 0.f, start_us + 30'005));
	pump(master, slave, start_us + 30'006);
	CHECK(slave.state() == airlink::LinkState::PairActive);

	airlink::SensorState sensor{};
	sensor.status_flags = airlink::AccelValid | airlink::GyroValid | airlink::AttitudeValid
			      | airlink::AngularVelocityValid | airlink::SampleFresh
			      | airlink::LocalRcValid | airlink::SafetyGateAllow
			      | airlink::AirlinkControlActive;
	sensor.attitude_q[0] = 32767;
	CHECK(slave.queue_sensor_state(sensor, start_us + 30'007));
	pump(slave, master, start_us + 30'008);
	CHECK(master.state() == airlink::LinkState::PairActive);

	// A duplicate PAIR_ARM must return the cached result and never ask the
	// wrapper to arm a second time after this epoch is already active.
	airlink::SessionActions duplicate_actions =
		slave.receive(arm_duplicate, start_us + 30'009);
	CHECK(!duplicate_actions.request_arm);
	CHECK(slave.state() == airlink::LinkState::PairActive);
}

void test_standby_waits_for_operator_and_revokes_on_safety_loss()
{
	auto master = std::make_unique<airlink::Session>(1);
	auto slave = std::make_unique<airlink::Session>(1);
	master->start(airlink::Role::Master, 1, ready_local(true), 1'000);
	slave->start(airlink::Role::Slave, 2, ready_local(false), 1'000);
	establish_ready(*master, *slave, 2'000);
	CHECK(master->request_control_enable(5'000));
	pump(*master, *slave, 5'001);
	pump(*slave, *master, 5'002);
	// Stay healthy for two seconds, far beyond the old 10 ms ACK expiry.
	for (uint64_t now = 6'000; now <= 2'006'000; now += 2'000) {
		exchange_heartbeat(*master, *slave, now);
		CHECK(master->state() == airlink::LinkState::PairControlReady);
		CHECK(slave->state() == airlink::LinkState::PairControlReady);
	}
	CHECK(master->request_pair_arm(2'006'001));
	pump(*master, *slave, 2'006'002);
	CHECK(slave->state() == airlink::LinkState::PairArmPending);

	// A separate authorization must be revoked when either endpoint loses RC safety.
	master = std::make_unique<airlink::Session>(1);
	slave = std::make_unique<airlink::Session>(1);
	master->start(airlink::Role::Master, 3, ready_local(true), 3'000'000);
	slave->start(airlink::Role::Slave, 4, ready_local(false), 3'000'000);
	establish_ready(*master, *slave, 3'001'000);
	CHECK(master->request_control_enable(3'004'000));
	pump(*master, *slave, 3'004'001);
	pump(*slave, *master, 3'004'002);
	auto unsafe = ready_local(false);
	unsafe.local_rc_valid = false;
	slave->set_local_status(unsafe);
	exchange_heartbeat(*master, *slave, 3'006'000);
	exchange_heartbeat(*master, *slave, 3'008'000);
	CHECK(master->state() == airlink::LinkState::Ready);
	CHECK(slave->state() == airlink::LinkState::Ready);
	CHECK(!master->request_pair_arm(3'008'001));

	// Authorization cannot survive loss of the periodic peer stream either.
	slave->set_local_status(ready_local(false));
	exchange_heartbeat(*master, *slave, 3'010'000);
	CHECK(master->request_control_enable(3'010'001));
	pump(*master, *slave, 3'010'002);
	pump(*slave, *master, 3'010'003);
	master->tick(3'030'000);
	slave->tick(3'030'000);
	CHECK(master->state() != airlink::LinkState::PairControlReady);
	CHECK(slave->state() != airlink::LinkState::PairControlReady);
	CHECK(!master->request_pair_arm(3'030'001));
}

void test_runtime_timing_does_not_refresh_periodic_safety()
{
	auto master = std::make_unique<airlink::Session>(1);
	auto slave = std::make_unique<airlink::Session>(1);
	master->start(airlink::Role::Master, 1, ready_local(true), 1'000);
	slave->start(airlink::Role::Slave, 2, ready_local(false), 1'000);
	establish_pair_active(*master, *slave, 2'000);
	pump(*slave, *master, 32'010); // drain the cached duplicate arm ACK
	const auto master_age = master->last_peer_age_us(33'000);
	const auto slave_age = slave->last_peer_age_us(33'000);
	master->queue_timing_probe(33'000);
	airlink::Frame probe{};
	CHECK(master->pop_tx(probe));
	CHECK(probe.header.message_id == static_cast<uint8_t>(airlink::MessageId::LinkTiming));
	CHECK(probe.header.payload_len == 8);
	slave->receive(probe, 33'500);
	pump(*slave, *master, 34'000);
	CHECK(master->timing_health().samples == 1);
	CHECK(master->timing_health().rtt_us == 1000);
	CHECK(master->last_peer_age_us(34'000) == master_age + 1000);
	CHECK(slave->last_peer_age_us(34'000) == slave_age + 1000);
	// Duplicate or wrong-session replies cannot produce a second RTT sample.
	slave->receive(probe, 34'100);
	pump(*slave, *master, 34'200);
	CHECK(master->timing_health().samples == 1);
	probe.header.source_session += 1;
	CHECK(!slave->receive(probe, 34'300).pair_active);
	CHECK(slave->tx_queue_size() == 0);
	CHECK(master->tick(45'000).request_disarm);
	CHECK(slave->tick(45'000).request_disarm);
	CHECK(master->pair_reattach_latched() && slave->pair_reattach_latched());
}

void test_estimator_kill_preempts_output_and_latches_both()
{
	auto master = std::make_unique<airlink::Session>(1);
	auto slave = std::make_unique<airlink::Session>(1);
	master->start(airlink::Role::Master, 1, ready_local(true), 1'000);
	slave->start(airlink::Role::Slave, 2, ready_local(false), 1'000);
	establish_pair_active(*master, *slave, 2'000);
	pump(*slave, *master, 32'010);
	airlink::ActuatorCommand command{};
	command.upper_motor = command.lower_motor = .8f;
	CHECK(master->queue_actuator_command(command, .8f, 33'000));
	const auto kill = master->fail_estimator(33'001);
	CHECK(kill.request_disarm && kill.pair_lost);
	CHECK(master->pair_reattach_latched());
	CHECK(!master->query_control_tx_allowed(33'002));
	CHECK(!master->queue_actuator_command(command, .8f, 33'002));
	airlink::Frame frame{};
	CHECK(master->pop_tx(frame));
	CHECK(frame.header.message_id == static_cast<uint8_t>(airlink::MessageId::EmergencyWithdraw));
	CHECK((frame.header.flags & airlink::Urgent) != 0);
	airlink::ReliableEvent event{};
	CHECK(airlink::decode_reliable_event(frame.payload, frame.header.payload_len, event));
	CHECK(event.reason_code == airlink::ReasonCode::EstimatorInvalid);
	CHECK(master->tx_queue_size() == 0); // Old high-thrust packet was discarded, not transmitted ahead of kill.
	const auto peer_kill = slave->receive(frame, 33'003);
	CHECK(peer_kill.request_disarm && peer_kill.clear_follower_control && peer_kill.pair_lost);
	CHECK(slave->pair_reattach_latched());
	pump(*slave, *master, 33'004);
	CHECK(!slave->receive(frame, 33'005).request_arm); // Duplicate kill is idempotent.
	master->set_armed(false); slave->set_armed(false);
	exchange_heartbeat(*master, *slave, 34'000);
	CHECK(master->pair_reattach_latched() && slave->pair_reattach_latched());
	CHECK(!master->request_control_enable(34'001));
	CHECK(master->reset_after_operator_disarm(35'000));
	CHECK(slave->pair_reattach_latched());
	CHECK(!master->request_control_enable(35'001));
	CHECK(slave->reset_after_operator_disarm(36'000));
	establish_pair_active(*master, *slave, 40'000);
}

void test_lost_kill_packet_still_stops_follower()
{
	auto master = std::make_unique<airlink::Session>(1);
	auto slave = std::make_unique<airlink::Session>(1);
	master->start(airlink::Role::Master, 1, ready_local(true), 1'000);
	slave->start(airlink::Role::Slave, 2, ready_local(false), 1'000);
	establish_pair_active(*master, *slave, 2'000);
	CHECK(master->fail_estimator(33'000).request_disarm);
	airlink::Frame dropped{};
	while (master->pop_tx(dropped)) {} // Simulate a lost emergency message.
	const auto fallback = slave->tick(43'001);
	CHECK(fallback.request_disarm && fallback.pair_lost && fallback.clear_follower_control);
	CHECK(slave->pair_reattach_latched());
}

void test_estimator_kill_cancels_pending_arm()
{
	auto master = std::make_unique<airlink::Session>(1);
	auto slave = std::make_unique<airlink::Session>(1);
	master->start(airlink::Role::Master, 1, ready_local(true), 1'000);
	slave->start(airlink::Role::Slave, 2, ready_local(false), 1'000);
	establish_ready(*master, *slave, 2'000);
	CHECK(master->request_control_enable(5'000));
	pump(*master, *slave, 5'001); pump(*slave, *master, 5'002);
	CHECK(master->request_pair_arm(6'000));
	airlink::Frame arm{}; CHECK(master->pop_tx(arm));
	CHECK(slave->receive(arm,6'001).request_arm);
	const auto kill = master->fail_estimator(6'002);
	CHECK(kill.pair_lost);
	airlink::Frame emergency{}; CHECK(master->pop_tx(emergency));
	CHECK(slave->receive(emergency,6'003).pair_lost);
	CHECK(!slave->notify_local_arm_result(true,6'004));
	pump(*slave,*master,6'005);
	CHECK(!master->notify_local_arm_result(true,6'006));
	CHECK(master->pair_reattach_latched() && slave->pair_reattach_latched());
}

void test_handshake_and_pair_lifecycle()
{
	constexpr uint32_t config_crc = 0x12345678u;
	auto master = std::make_unique<airlink::Session>(config_crc);
	auto slave = std::make_unique<airlink::Session>(config_crc);
	master->start(airlink::Role::Master, 0x11111111u, ready_local(true), 1'000);
	slave->start(airlink::Role::Slave, 0x22222222u, ready_local(false), 1'000);
	establish_pair_active(*master, *slave, 2'000);
	CHECK(master->peer_session_id() == slave->session_id());
	CHECK(slave->peer_session_id() == master->session_id());

	const airlink::SessionActions master_lost = master->tick(50'000);
	const airlink::SessionActions slave_lost = slave->tick(50'000);
	CHECK(master->state() == airlink::LinkState::PairLost);
	CHECK(slave->state() == airlink::LinkState::PairLost);
	CHECK(master_lost.pair_lost);
	CHECK(slave_lost.pair_lost);
	CHECK(master_lost.request_disarm);
	CHECK(slave_lost.request_disarm);
	CHECK(master->pair_reattach_latched());
	CHECK(slave->pair_reattach_latched());

	master->set_local_status(ready_local(true));
	slave->set_local_status(ready_local(false));
	CHECK(!master->request_control_enable(50'001));
}

void test_handshake_config_mismatch()
{
	auto master = std::make_unique<airlink::Session>(0x11111111u);
	auto slave = std::make_unique<airlink::Session>(0x22222222u);
	master->start(airlink::Role::Master, 10, ready_local(true), 10'000);
	slave->start(airlink::Role::Slave, 20, ready_local(false), 10'000);
	exchange_heartbeat(*master, *slave, 11'000);
	pump(*master, *slave, 11'001);
	pump(*slave, *master, 11'002);
	CHECK(master->state() == airlink::LinkState::Discovered);
	CHECK(slave->state() == airlink::LinkState::HandshakePending);
	CHECK(!master->request_control_enable(11'003));
}

void test_same_role_never_links()
{
	auto first = std::make_unique<airlink::Session>(1);
	auto second = std::make_unique<airlink::Session>(1);
	first->start(airlink::Role::Master, 10, ready_local(true), 20'000);
	second->start(airlink::Role::Master, 20, ready_local(true), 20'000);
	exchange_heartbeat(*first, *second, 21'000);
	CHECK(first->state() == airlink::LinkState::Down);
	CHECK(second->state() == airlink::LinkState::Down);
	CHECK(first->peer_session_id() == 0);
	CHECK(second->peer_session_id() == 0);
}

void test_unknown_command_gets_unsupported_ack()
{
	auto master = std::make_unique<airlink::Session>(1);
	auto slave = std::make_unique<airlink::Session>(1);
	master->start(airlink::Role::Master, 10, ready_local(true), 25'000);
	slave->start(airlink::Role::Slave, 20, ready_local(false), 25'000);
	establish_ready(*master, *slave, 26'000);

	uint8_t payload[8] {0, 0, 0, 0, 0x44, 0x33, 0x22, 0x11};
	uint8_t encoded[airlink::kMaxFrameSize] {};
	size_t frame_len = 0;
	airlink::Header header{};
	header.frame_type = airlink::FrameType::Command;
	header.message_id = 0x20;
	header.flags = airlink::AckRequired;
	header.source_role = airlink::Role::Master;
	header.dest_role = airlink::Role::Slave;
	header.source_session = master->session_id();
	header.sequence = 123;
	header.payload_len = sizeof(payload);
	CHECK(airlink::build_frame(header, payload, sizeof(payload), encoded, sizeof(encoded),
				   frame_len));
	airlink::Frame frame{};
	airlink::StreamParser parser;
	parser.feed(encoded, frame_len, 28'000);
	CHECK(parser.next(frame));
	airlink::Ack expected_ack{};
	expected_ack.target_peer_session = master->session_id();
	expected_ack.event_id = 0x11223344u;
	expected_ack.acked_message_id = 0x20;
	expected_ack.result = airlink::AckResult::Unsupported;
	uint8_t ack_payload[airlink::kAckPayloadSize] {};
	CHECK(airlink::encode_ack(expected_ack, ack_payload));
	airlink::Header ack_header{};
	ack_header.frame_type = airlink::FrameType::Command;
	ack_header.flags = airlink::Response;
	ack_header.message_id = static_cast<uint8_t>(airlink::MessageId::Ack);
	ack_header.source_role = airlink::Role::Slave;
	ack_header.dest_role = airlink::Role::Master;
	ack_header.source_session = slave->session_id();
	ack_header.sequence = 1;
	ack_header.payload_len = sizeof(ack_payload);
	uint8_t ack_encoded[airlink::kMaxFrameSize] {};
	size_t ack_len = 0;
	CHECK(airlink::build_frame(ack_header, ack_payload, sizeof(ack_payload), ack_encoded,
				   sizeof(ack_encoded), ack_len));
	(void)slave->receive(frame, 28'000);
	CHECK(slave->unknown_messages() == 1);
	CHECK(slave->tx_queue_size() == 1);
	airlink::Frame ack_frame{};
	CHECK(slave->pop_tx(ack_frame));
	CHECK(ack_frame.header.message_id == static_cast<uint8_t>(airlink::MessageId::Ack));
	airlink::Ack ack{};
	CHECK(airlink::decode_ack(ack_frame.payload, ack_frame.header.payload_len, ack));
	CHECK(ack.result == airlink::AckResult::Unsupported);
	CHECK(ack.acked_message_id == 0x20);
	CHECK(ack.event_id == 0x11223344u);
}

void test_reliable_retry_limit()
{
	auto master = std::make_unique<airlink::Session>(1);
	auto slave = std::make_unique<airlink::Session>(1);
	master->start(airlink::Role::Master, 10, ready_local(true), 30'000);
	slave->start(airlink::Role::Slave, 20, ready_local(false), 30'000);
	exchange_heartbeat(*master, *slave, 31'000);
	uint64_t now = 31'001;
	int sends = 0;

	for (int iteration = 0; iteration < 6; ++iteration) {
		airlink::Frame frame{};

		while (master->pop_tx(frame)) {
			if (frame.header.message_id == static_cast<uint8_t>(airlink::MessageId::LinkHandshake)) {
				++sends;

				if (sends > 1) {
					CHECK((frame.header.flags & airlink::Retry) != 0);
				}
			}
		}

		now += 5'000;
		(void)master->tick(now);
	}

	CHECK(sends >= 4);
	CHECK(sends <= 5);
	CHECK(master->state() == airlink::LinkState::Discovered);
}

void test_link_test_round_trip()
{
	auto master = std::make_unique<airlink::Session>(1);
	auto slave = std::make_unique<airlink::Session>(1);
	master->start(airlink::Role::Master, 10, ready_local(true), 40'000);
	slave->start(airlink::Role::Slave, 20, ready_local(false), 40'000);
	establish_ready(*master, *slave, 41'000);
	airlink::LinkTestRequest request{};
	request.payload_len = 64;
	request.rate_hz = 500;
	request.duration_ms = 1000;
	CHECK(master->start_link_test(request, 61'000));
	pump(*master, *slave, 61'001);
	pump(*slave, *master, 61'002);
	CHECK(master->link_test_active());
	CHECK(slave->link_test_active());

	uint64_t now = 61'003;
	bool finished = false;

	for (int iteration = 0; iteration < 5000; ++iteration) {
		master->tick(now);
		pump(*master, *slave, now);
		slave->tick(now);
		pump(*slave, *master, now + 1);
		now += 2000;

		if (!master->link_test_active()) {
			finished = true;
			break;
		}
	}

	CHECK(finished);
	CHECK(master->link_test_stats().sent > 400);
	CHECK(master->link_test_stats().echoed > 400);
	CHECK(master->link_test_stats().missing == 0);
	CHECK(master->link_test_stats().offered_bytes_per_second > 0);
}

void test_v2_preheat_and_yaw_reference()
{
	auto master = std::make_unique<airlink::Session>(0x12345678u);
	auto slave = std::make_unique<airlink::Session>(0x12345678u);
	master->start(airlink::Role::Master, 0x11111111u, ready_local(true), 80'000);
	slave->start(airlink::Role::Slave, 0x22222222u, ready_local(false), 80'000);
	establish_ready(*master, *slave, 81'000);

	airlink::SensorState sensor{};
	sensor.status_flags = airlink::AccelValid | airlink::GyroValid
			      | airlink::AttitudeValid | airlink::AngularVelocityValid
			      | airlink::SampleFresh;
	sensor.imu_sample_time_us = 82'000;
	sensor.attitude_sample_time_us = 82'000;
	sensor.attitude_q[0] = 32767;
	CHECK(slave->queue_sensor_state(sensor, 83'000));
	pump(*slave, *master, 83'001);
	CHECK(master->state() == airlink::LinkState::Ready);
	CHECK(master->last_peer_sensor_arrival_us() == 83'001);
	CHECK(master->last_peer_sensor_sequence() != 0);

	airlink::YawReference reference{};
	reference.source_sample_time_us = 0x89ABCDEFu;
	reference.yaw_rad = 0.25f;
	reference.yaw_sigma_rad = 0.1f;
	reference.reset_counter = 3;
	reference.quality = 2;
	reference.flags = airlink::YawValid;
	CHECK(master->queue_yaw_reference(reference, 84'000));
	pump(*master, *slave, 84'001);
	CHECK(slave->last_peer_yaw_arrival_us() == 84'001);
	CHECK(slave->last_peer_yaw_sequence() != 0);

	reference.flags = airlink::ReservedYawReferenceFlags;
	CHECK(!master->queue_yaw_reference(reference, 85'000));
}

void test_unconnected_and_restarted_peer()
{
	auto master = std::make_unique<airlink::Session>(1);
	auto slave = std::make_unique<airlink::Session>(1);
	master->start(airlink::Role::Master, 1, ready_local(true), 1'000);
	slave->start(airlink::Role::Slave, 2, ready_local(false), 1'000);
	CHECK(!master->tick(1'000'000).pair_lost);
	CHECK(!master->pair_reattach_latched());
	establish_ready(*master, *slave, 1'002'000);
	slave->start(airlink::Role::Slave, 3, ready_local(false), 1'006'000);
	slave->tick(1'006'000);
	airlink::Frame frame{};
	bool lost = false;
	while (slave->pop_tx(frame)) {
		lost |= master->receive(frame, 1'006'001).pair_lost;
	}
	CHECK(lost && master->pair_reattach_latched());
	// Restarting a session on the same endpoint cannot clear its fault latch.
	master->start(airlink::Role::Master, 4, ready_local(true), 1'008'000);
	CHECK(master->pair_reattach_latched());
}

void test_ground_connection_fault_lock()
{
	for (bool airborne : {false, true}) {
		auto master = std::make_unique<airlink::Session>(1);
		auto slave = std::make_unique<airlink::Session>(1);
		master->start(airlink::Role::Master, 1, ready_local(true), 1'000);
		slave->start(airlink::Role::Slave, 2, ready_local(false), 1'000);
		if (airborne) {
			establish_pair_active(*master, *slave, 2'000);
			auto local = ready_local(true);
			local.armed = true;
			local.airborne_latched = true;
			master->set_local_status(local);
			slave->set_local_status(local);
		} else {
			establish_ready(*master, *slave, 2'000);
		}
		const auto a = master->tick(50'000);
		const auto b = slave->tick(50'000);
		CHECK(a.pair_lost && b.pair_lost);
		CHECK(master->pair_reattach_latched() && slave->pair_reattach_latched());
		if (airborne) {
			CHECK(a.request_disarm && b.request_disarm);
		}
		master->set_local_status(ready_local(true));
		slave->set_local_status(ready_local(false));
		exchange_heartbeat(*master, *slave, 51'000);
		CHECK(!master->request_control_enable(51'001));
		CHECK(master->pair_reattach_latched() && slave->pair_reattach_latched());
	}
}

void test_single_direction_disconnect()
{
	for (bool drop_master_tx : {false, true}) {
		auto master = std::make_unique<airlink::Session>(1);
		auto slave = std::make_unique<airlink::Session>(1);
		master->start(airlink::Role::Master, 1, ready_local(true), 1'000);
		slave->start(airlink::Role::Slave, 2, ready_local(false), 1'000);
		establish_ready(*master, *slave, 2'000);
		for (uint64_t now = 6'000; now <= 40'000; now += 2'000) {
			master->tick(now);
			slave->tick(now);
			airlink::Frame frame{};
			while (master->pop_tx(frame)) {
				if (!drop_master_tx) { slave->receive(frame, now); }
			}
			while (slave->pop_tx(frame)) {
				if (drop_master_tx) { master->receive(frame, now); }
			}
		}
		CHECK(master->pair_reattach_latched());
		CHECK(slave->pair_reattach_latched());
	}
}

void test_disarm_cancels_delayed_arm_ack()
{
	auto master = std::make_unique<airlink::Session>(1);
	auto slave = std::make_unique<airlink::Session>(1);
	master->start(airlink::Role::Master, 1, ready_local(true), 1'000);
	slave->start(airlink::Role::Slave, 2, ready_local(false), 1'000);
	establish_ready(*master, *slave, 2'000);
	CHECK(master->request_control_enable(5'000));
	pump(*master, *slave, 5'001);
	pump(*slave, *master, 5'002);
	CHECK(master->request_pair_arm(6'000));
	pump(*master, *slave, 6'001);
	CHECK(slave->notify_local_arm_result(true, 6'002));
	airlink::Frame delayed_ack{};
	CHECK(slave->pop_tx(delayed_ack));
	CHECK(slave->request_pair_disarm(airlink::ReasonCode::OperatorRequest, 6'003));
	pump(*slave, *master, 6'004);
	CHECK(!master->receive(delayed_ack, 6'005).request_arm);
	CHECK(master->state() == airlink::LinkState::Ready);
	CHECK(!master->pair_reattach_latched());
}

void test_rc_disarm_from_either_endpoint()
{
	for (bool slave_disarms : {false, true}) {
		auto master = std::make_unique<airlink::Session>(1);
		auto slave = std::make_unique<airlink::Session>(1);
		master->start(airlink::Role::Master, 1, ready_local(true), 1'000);
		slave->start(airlink::Role::Slave, 2, ready_local(false), 1'000);
		establish_pair_active(*master, *slave, 2'000);
		auto &sender = slave_disarms ? *slave : *master;
		auto &receiver = slave_disarms ? *master : *slave;
		CHECK(sender.request_pair_disarm(airlink::ReasonCode::OperatorRequest, 33'000));
		airlink::Frame frame{};
		bool disarmed = false;
		while (sender.pop_tx(frame)) {
			disarmed |= receiver.receive(frame, 33'001).request_disarm;
		}
		CHECK(disarmed);
		master->set_local_status(ready_local(true));
		slave->set_local_status(ready_local(false));
		pump(receiver, sender, 33'002);
		exchange_heartbeat(*master, *slave, 34'000);
		CHECK(!master->pair_reattach_latched() && !slave->pair_reattach_latched());
		CHECK(master->state() == airlink::LinkState::Ready);
		CHECK(slave->state() == airlink::LinkState::Ready);
		CHECK(master->request_control_enable(34'001));
	}
}

void test_rc_disarm_recovers_fault()
{
	auto master = std::make_unique<airlink::Session>(1);
	auto slave = std::make_unique<airlink::Session>(1);
	master->start(airlink::Role::Master, 1, ready_local(true), 1'000);
	slave->start(airlink::Role::Slave, 2, ready_local(false), 1'000);
	establish_pair_active(*master, *slave, 2'000);
	CHECK(master->tick(60'000).request_disarm);
	CHECK(slave->tick(60'000).request_disarm);
	CHECK(master->pair_reattach_latched() && slave->pair_reattach_latched());
	CHECK(!master->reset_after_operator_disarm(60'001)); // Still armed.
	master->set_armed(false);
	slave->set_armed(false);
	CHECK(!master->request_control_enable(60'002)); // Automatic disarm does not clear fault.
	CHECK(master->reset_after_operator_disarm(61'000));
	CHECK(!master->pair_reattach_latched());
	CHECK(slave->pair_reattach_latched()); // Each endpoint requires its own RC acknowledgement.
	exchange_heartbeat(*master, *slave, 61'100);
	CHECK(!master->pair_reattach_latched());
	CHECK(!master->request_control_enable(61'101));
	CHECK(slave->reset_after_operator_disarm(62'000));
	CHECK(!slave->pair_reattach_latched());
	CHECK(master->state() == airlink::LinkState::Discovered);
	CHECK(slave->state() == airlink::LinkState::Down);
	establish_pair_active(*master, *slave, 70'000);
}

} // namespace

int main()
{
	test_estimator_kill_preempts_output_and_latches_both();
	test_lost_kill_packet_still_stops_follower();
	test_estimator_kill_cancels_pending_arm();
	test_runtime_timing_does_not_refresh_periodic_safety();
	test_standby_waits_for_operator_and_revokes_on_safety_loss();
	test_rc_disarm_recovers_fault();
	test_unconnected_and_restarted_peer();
	test_ground_connection_fault_lock();
	test_single_direction_disconnect();
	test_disarm_cancels_delayed_arm_ack();
	test_rc_disarm_from_either_endpoint();
	test_handshake_and_pair_lifecycle();
	test_handshake_config_mismatch();
	test_same_role_never_links();
	test_unknown_command_gets_unsupported_ack();
	test_reliable_retry_limit();
	test_link_test_round_trip();
	test_v2_preheat_and_yaw_reference();

	if (g_failures != 0) {
		std::fprintf(stderr, "%d AIRLINK session check(s) failed\n", g_failures);
		return EXIT_FAILURE;
	}

	std::puts("AIRLINK session checks passed");
	return EXIT_SUCCESS;
}
