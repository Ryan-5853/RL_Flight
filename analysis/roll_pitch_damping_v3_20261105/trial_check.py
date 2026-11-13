"""Nonlinear rigid-body + actuator check of the actual C++ implementation.
Includes 3-axis rotation, translation, rotor and servo first-order dynamics,
grid force/arm/yaw coupling, and constant roll disturbance. No measured wake,
contact, battery, IMU noise, linkage backlash, or hardware validation.
"""
from pathlib import Path
import os
import ctypes,tempfile,subprocess,json
import numpy as np,yaml
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
cfg=yaml.safe_load((ROOT/'SimEnv/configs/sim2real_micro_coaxial_3100kv_airframe1_manual_v2.yaml').read_text())
table=np.array(cfg['motors'][0]['pwm_to_rpm_table']['value']);kt=np.array(cfg['aerodynamics']['thrust_coefficients']['value']);arms=np.array([g['aerodynamic_center_b']['value'] for g in cfg['aerodynamics']['grids']]);axes=np.array([g['deflection_axis_b']['value'] for g in cfg['aerodynamics']['grids']]);parts=np.array([.1166666667,.1166666667,.1166666666]);I=np.array([.0005,.0005,.0007333333]);mass=.66;dt=.002;g=9.80665
source=r'''
#include "LqiCoaxialCollectiveCore.hpp"
#include "LqiCoaxialCollectivePayload.hpp"
#include "LqiManualReference.hpp"
static float tuned_gain[24]{};
static LqiCoaxialCollectiveModel tuned_model = lqi_coaxial_collective::kModel;
static bool prepare() {
 for(int i=0;i<24;++i) { tuned_gain[i]=lqi_coaxial_collective::kAttitudeGain[i]; }
 for(int axis=0;axis<2;++axis) {
  for(int i=0;i<8;++i) { tuned_gain[axis*8+i]=0; }
  tuned_gain[axis*8+axis]=.012f;tuned_gain[axis*8+2+axis]=.004f;tuned_gain[axis*8+5+axis]=.02f;
 }
 tuned_model.attitude_gain=tuned_gain;tuned_model.roll_pitch_integral_moment_fraction=.9f;
 return true;
}
static bool prepared = prepare();
static LqiCoaxialCollectiveCore core{tuned_model};
static LqiCoaxialManualReference reference{lqi_coaxial_collective::kModel};
extern "C" void reset() { core.reset();reference.reset(); }
extern "C" int tick(const float *x,float *out) {
 LqiCoaxialCollectiveCore::Input in{};LqiCoaxialCollectiveCore::ControlOutput y{};
 if(!reference.build(x,x+4,x[10],x[11],x[12],x[13],in)) { return 0; }
 for(int i=0;i<3;++i) { in.acceleration_b[i]=x[7+i]; }
 for(int axis=0;axis<2;++axis) {
  const auto angle=[](const float *q,int a) {
   if(a==0) { return std::atan2(2*(q[0]*q[1]+q[2]*q[3]),1-2*(q[1]*q[1]+q[2]*q[2])); }
   float v=2*(q[0]*q[2]-q[3]*q[1]);return std::asin(std::fmax(-1.f,std::fmin(1.f,v)));
  };
  float error=std::fabs(angle(in.attitude_q_wb,axis)-angle(in.target_attitude_q_wb,axis));
  float blend=std::fmax(0.f,std::fmin(1.f,(error-.104719755f)/.104719755f));
  blend=blend*blend*(3-2*blend);tuned_gain[axis*8+axis]=.012f+(.060f-.012f)*blend;
 }
 if(!core.step(in,y)) { return 0; }
 out[0]=y.upper;out[1]=y.lower;
 for(int i=0;i<3;++i) { out[2+i]=y.servos[i];out[5+i]=core.persistent()[2+i]; }
 return 1;
}
'''
def rotation(q):
 w,x,y,z=q
 return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
def multiply(q,p):
 w,x,y,z=q;a,b,c,d=p
 return np.array([w*a-x*b-y*c-z*d,w*b+x*a+y*d-z*c,w*c-x*d+y*a+z*b,w*d+x*c-y*b+z*a])
cases=[('nominal',.7,.06,1.,0.),('roll_bias',.7,.06,1.,.006),('slower_servo',.7,.10,1.,.004),('stronger_grid',.7,.12,3.,.004),('weaker_lower',.65,.06,1.,.004),('stronger_lower',.8,.06,1.,.004)]
results=[]
with tempfile.TemporaryDirectory(prefix='coupled-yaw-') as work:
 work=Path(work);(work/'core.cpp').write_text(source);subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC','-I',str(ROOT/'px4_trans/px4/src/modules/nn_control'),str(work/'core.cpp'),'-o',str(work/'core.so')],check=True)
 lib=ctypes.CDLL(str(work/'core.so'));fp=ctypes.POINTER(ctypes.c_float);lib.tick.argtypes=[fp,fp];lib.tick.restype=ctypes.c_int
 for label,rho,tau,vector_gain,roll_bias in cases:
  lib.reset();q=Rotation.from_euler('xyz',[3,-2,0],degrees=True).as_quat()[[3,0,1,2]];omega=np.zeros(3);speed=np.array([963.975,1041.093]);servo=np.ones(3)*-.08;held=servo.copy();pending=[servo.copy() for _ in range(3)];acc=np.array([0,0,-g]);velocity=np.zeros(3);position=np.zeros(3);records=[]
  for n in range(8000):
   t=n*dt;rc=np.zeros(4)
   if 2<=t<5:rc[0]=.4
   elif 5<=t<8:rc[0]=-.4;rc[1]=.3
   elif 8<=t<11:rc[1]=-.3;rc[2]=.5
   elif 11<=t<13:rc[2]=-.5
   x=np.r_[q,omega,acc,rc].astype('f4');out=np.zeros(8,dtype='f4');assert lib.tick(x.ctypes.data_as(fp),out.ctypes.data_as(fp))
   speed+=(np.interp(out[:2],table[:,0],table[:,1])-speed)*(1-np.exp(-dt/np.array([.04,.05])))
   pending.append(out[2:5].copy());pending.pop(0)
   if n % 10 == 0:
    held=pending[0].copy()
   servo+=(held-servo)*(1-np.exp(-dt/tau))
   T=kt[0]*speed[0]**2+kt[1]*speed[1]**2+kt[2]*speed.prod()
   attenuation=1-.15*abs(servo);effective=attenuation-.15*((1-attenuation).sum()-(1-attenuation));angle=.245*vector_gain*servo
   neutral=np.array([0,0,-1.]);directions=np.cos(angle)[:,None]*neutral+np.sin(angle)[:,None]*np.cross(axes,neutral)
   grid_force=T*parts[:,None]*effective[:,None]*directions
   force=T*.65*neutral+grid_force.sum(0)
   moment=np.cross(arms,grid_force).sum(0)+np.array([roll_bias,0,2.3e-8*(speed[0]**2-rho*speed[1]**2)])
   omega+=((moment-np.cross(omega,I*omega))/I)*dt
   theta=np.linalg.norm(omega)*dt
   dq=np.r_[np.cos(theta/2),omega*dt*.5] if theta<1e-8 else np.r_[np.cos(theta/2),omega/np.linalg.norm(omega)*np.sin(theta/2)]
   q=multiply(q,dq);q/=np.linalg.norm(q);acc=force/mass
   velocity+=(rotation(q)@acc+np.array([0,0,g]))*dt;position+=velocity*dt
   euler=Rotation.from_quat(q[[1,2,3,0]]).as_euler('xyz',degrees=True)
   predicted=np.interp(out[:2],table[:,0],table[:,1]);ratio=max(predicted)/max(min(predicted),1e-9)
   assert max(out[:2])<=.900001 and ratio<=1.15001
   assert max(abs(out[5:7]*.02))<=.008101
   records.append([t,*euler,*omega,*out[:5],*out[5:8],ratio,*position])
   if max(abs(euler[:2]))>45:break
  a=np.array(records);last=a[:,0]>=a[-1,0]-1
  passed=len(a)==8000 and abs(a[:,1:3]).max()<30 and abs(a[last,1:3].mean(0)).max()<6 and abs(a[last,6].mean())<.2
  results.append({'case':label,'assumed_lower_torque_ratio':rho,'servo_tau_s':tau,'grid_vector_gain_multiplier':vector_gain,'constant_roll_disturbance_n_m':roll_bias,'duration_s':float(a[-1,0]),'model_pwm_hold_hz':50,'extra_delay_s':.004,'max_abs_roll_pitch_deg':abs(a[:,1:3]).max(0).tolist(),'final_mean_roll_pitch_deg':a[last,1:3].mean(0).tolist(),'final_mean_yaw_rate':float(a[last,6].mean()),'pass_declared_model_checks':bool(passed)})
  np.savetxt(OUT/('trial_'+label+'.csv'),a,delimiter=',',header='t,roll,pitch,yaw,p,q,r,upper,lower,servo1,servo2,servo3,Iroll,Ipitch,Iyaw,pred_ratio,x,y,z',comments='')
(OUT/'trial_coupled_metrics.json').write_text(json.dumps({'scope':__doc__,'results':results},indent=2));print(json.dumps(results,indent=2))
assert all(r['pass_declared_model_checks'] for r in results),'some model cases failed; inspect before packaging'
