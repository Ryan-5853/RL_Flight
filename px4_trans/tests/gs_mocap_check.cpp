#include "../px4/src/modules/gs_link/GsMocap.hpp"
#include <cassert>
#include <limits>

int main()
{
	gs_link::MocapAttitudePayload sample{7, 10, 990000, {1.f, 0.f, 0.f, 0.f}, 0.0025f};
	const auto valid = [&](const gs_link::MocapAttitudePayload &s) {
		return gs_link::validMocapSample(s, 1000000, 7, true, 7, 9, 980000);
	};
	assert(valid(sample));
	assert(!gs_link::validMocapSample(sample, 1000000, 7, false, 7, 9, 980000));
	assert(!gs_link::validMocapSample(sample, 1000000, 8, true, 7, 9, 980000));
	assert(!gs_link::validMocapSample(sample, 1000000, 7, true, 7, 10, 980000));
	auto changed = sample;
	changed.sample_sequence = 8;
	assert(!valid(changed));
	changed = sample; changed.sample_time_us = 899999;
	assert(!valid(changed));
	changed.sample_time_us = 1010001;
	assert(!valid(changed));
	changed.sample_time_us = 1005000;
	assert(valid(changed)); // midpoint clock error within the accepted RTT bound
	changed.sample_time_us = 980000;
	assert(!valid(changed));
	changed = sample; changed.q[0] = 0.f;
	assert(!valid(changed));
	changed.q[0] = std::numeric_limits<float>::quiet_NaN();
	assert(!valid(changed));
	changed = sample; changed.variance = -1.f;
	assert(!valid(changed));
	changed.variance = std::numeric_limits<float>::infinity();
	assert(!valid(changed));
	changed = sample; changed.q[0] = -1.f;
	assert(valid(changed));
	changed.sample_sequence = 0;
	assert(gs_link::validMocapSample(changed, 1000000, 7, true, 7, UINT32_MAX, 980000));
	assert(gs_link::validMocapSample(changed, 1000000, 7, true, 6, 100, 980000));
}
