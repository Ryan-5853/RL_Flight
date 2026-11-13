#include "../px4/src/modules/nn_control/LqiCoaxialCollectiveCore.hpp"
#ifndef COAXIAL_PAYLOAD_HEADER
#define COAXIAL_PAYLOAD_HEADER "../px4/src/modules/nn_control/LqiCoaxialCollectivePayload.hpp"
#endif
#include COAXIAL_PAYLOAD_HEADER
#include "../px4/src/modules/nn_control/LqiManualReference.hpp"
#include <cmath>
#include <cstdio>

using M = LqiCoaxialCollectiveModel;
using Core = LqiCoaxialCollectiveCore;
static int failures = 0;
static void check(bool condition, const char *label) { if (!condition) { ++failures; std::printf("FAIL %s\n",label); } }
static Core::Input input_for(int axis, float angle, float rate)
{
	Core::Input in{};
	in.attitude_q_wb[0] = std::cos(angle/2);
	in.attitude_q_wb[axis+1] = std::sin(angle/2);
	in.target_attitude_q_wb[0] = 1;
	in.angular_velocity_b[axis] = rate;
	in.acceleration_b[2] = -9.80665f;
	in.collective_base = .5f;
	return in;
}
static float common(const Core::ControlOutput &out) { return (out.servos[0]+out.servos[1]+out.servos[2])/3; }
int main()
{
	const auto &model = lqi_coaxial_collective::kModel;
	check(model.collective_contract == M::CollectiveContract::PilotAccelerationSharedYawAntiWindup,"revision 2 contract active");
	for (int axis=0;axis<2;++axis) {
		const float gain=model.attitude_gain[axis*8+5+axis];
		const float limit=model.maximum_moment_n_m[axis]*model.roll_pitch_integral_moment_fraction;
		for (int sign=-1;sign<=1;sign+=2) {
			float state[5]{},next[5]{};Core::ControlOutput out{};
			const float clipping_angle=2*model.maximum_moment_n_m[axis]/model.attitude_gain[axis*8+axis];
			auto in=input_for(axis,sign*clipping_angle,0);
			for (int i=0;i<10000;++i) {
				check(Core::stepFromState(model,in,state,out,next),"long moment-clipped run valid");
				for(int j=0;j<5;++j) { state[j]=next[j]; }
			}
			check(std::fabs(state[axis+2]*gain)<=limit+1e-7f,"RP integral hard bound after 20 seconds");
			check(std::fabs(state[axis+2])<1e-5f,"RP integral does not build into existing P saturation");

			state[axis+2]=sign*.04f;
			const float clipping_rate=2*model.maximum_moment_n_m[axis]/model.attitude_gain[axis*8+2+axis];
			in=input_for(axis,-sign*.01f,sign*clipping_rate);
			check(Core::stepFromState(model,in,state,out,next),"unwind run valid");
			check(sign*(next[axis+2]-state[axis+2])<0,"RP may unwind while rate demand keeps moment saturated");
			in=input_for(axis,sign*.01f,sign*clipping_rate);
			check(Core::stepFromState(model,in,state,out,next),"blocked integral run valid");
			check(std::fabs(next[axis+2]-state[axis+2])<1e-6f,"RP cannot integrate farther into moment clipping");
			state[axis+2]=sign*1000.f;
			check(Core::stepFromState(model,in,state,out,next),"oversized inherited integral valid");
			check(std::fabs(next[axis+2]*gain)<=limit+1e-7f,"oversized inherited state clipped even when blocked");
		}
	}
	{
		float no_grid_policy[M::YawBalanceSize];
		for(int i=0;i<M::YawBalanceSize;++i) { no_grid_policy[i]=model.yaw_balance[i]; }
		no_grid_policy[M::GridCommonCommand]=0;
		M no_grid=model;no_grid.yaw_balance=no_grid_policy;
		for(int axis=0;axis<2;++axis) {
			float state[5]{},next[5]{};Core::ControlOutput out{};state[axis+2]=.04f;
			const float clipping_rate=2*model.maximum_moment_n_m[axis]/model.attitude_gain[axis*8+2+axis];
			auto in=input_for(axis,-.01f,clipping_rate);in.angular_velocity_b[1-axis]=clipping_rate;in.collective_base=.3f;
			check(Core::stepFromState(no_grid,in,state,out,next),"servo-saturated unwind valid");
			check(std::fmax(std::fabs(out.servos[0]),std::fmax(std::fabs(out.servos[1]),std::fabs(out.servos[2])))>.999f,"scenario reaches servo vector scaling");
			check(next[axis+2]<state[axis+2],"RP may unwind through actual servo scaling saturation");
		}
	}
	{
		float state[5]{},next[5]{};Core::ControlOutput out{};
		auto in=input_for(0,.05f,0);in.collective_base=.1f;
		check(Core::stepFromState(model,in,state,out,next)&&next[2]==0&&next[3]==0,"no RP integral accumulation at low thrust");
	}
	{
		float state[5]{},next[5]{};Core::ControlOutput out{};
		auto in=input_for(0,0,0);in.angular_velocity_b[2]=-2.f;
		check(Core::stepFromState(model,in,state,out,next),"positive yaw shortage valid");
		check(std::fabs(common(out))<1e-5f,"negative grid common retreats to zero for positive yaw");
		check(lqi_coaxial_collective_detail::feasible_motors(model,out.upper,out.lower),"retreat preserves rotor envelope");
		in.angular_velocity_b[2]=2.f;
		check(Core::stepFromState(model,in,state,out,next),"negative yaw shortage valid");
		check(common(out)<-.159f&&common(out)>=-.16001f,"negative yaw uses only bounded existing common reserve");
		check(lqi_coaxial_collective_detail::feasible_motors(model,out.upper,out.lower),"shared allocation preserves rotor envelope");
	}
	{
		// Shared yaw is piecewise continuous, avoiding a whole-bias on/off relay.
		float state[5]{},next[5]{};Core::ControlOutput out{};float last=0;
		for(int i=0;i<=1000;++i) {
			auto in=input_for(0,0,0);in.angular_velocity_b[2]=-.5f+i*.001f;
			check(Core::stepFromState(model,in,state,out,next),"continuous yaw sweep valid");
			if(i) { check(std::fabs(common(out)-last)<.005f,"common continuous across allocation boundary"); }
			last=common(out);
		}
	}
	{
		LqiCoaxialManualReference reference{model};Core::Input in{};
		float q[4]{1,0,0,0},rates[3]{};float first[4]{};
		for(int i=0;i<2000;++i) {
			check(reference.build(q,rates,.3f,-.2f,1,0,in),"rate reference builds");
			if(!i) { for(int j=0;j<4;++j) { first[j]=in.target_attitude_q_wb[j]; } }
			for(int j=0;j<4;++j) { check(std::fabs(in.target_attitude_q_wb[j]-first[j])<1e-6f,"untracked yaw does not rotate RP reference"); }
		}
		q[0]=std::cos(.5f);q[3]=std::sin(.5f);
		check(reference.build(q,rates,0,0,1,0,in),"heading update builds");
		check(std::fabs(in.target_attitude_q_wb[0]-q[0])<1e-6f&&std::fabs(in.target_attitude_q_wb[3]-q[3])<1e-6f,"RP reference follows actual heading");
	}
	std::printf("shared yaw / RP anti-windup checks: %d failures\n",failures);
	return failures!=0;
}
