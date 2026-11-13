"""Reduced nonlinear yaw/vertical sensitivity check using the actual C++ core.
Not a 6-DoF flight simulation: no rotor acceleration reaction torque, contact,
translational wake, sensor errors, or real battery/motor-speed feedback.
"""
from pathlib import Path
import ctypes,json,subprocess,tempfile
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
core=ROOT/'analysis/crash_20261105_unknown/controller_v1_source'
config=yaml.safe_load((ROOT/'SimEnv/configs/sim2real_micro_coaxial_3100kv_airframe1_manual_v2.yaml').read_text())
table=np.array(config['motors'][0]['pwm_to_rpm_table']['value'])
kt=np.array(config['aerodynamics']['thrust_coefficients']['value']);kq=2.3e-8
inertia=config['body']['inertia_diagonal_b']['value'][2];mass=.66;g=9.80665;dt=.002
wrapper=r'''
#include "LqiCoaxialCollectiveCore.hpp"
#include "LqiCoaxialCollectivePayload.hpp"
extern "C" int tick(float rate, float target, float az, float *persistent, float *output) {
 LqiCoaxialCollectiveCore::Input input{};
 input.attitude_q_wb[0] = input.target_attitude_q_wb[0] = 1;
 input.angular_velocity_b[2] = rate; input.target_angular_velocity_b[2] = target;
 input.acceleration_b[2] = az; input.collective_base = .5f;
 LqiCoaxialCollectiveCore::ControlOutput out{}; float next[5]{};
 if (!LqiCoaxialCollectiveCore::stepFromState(lqi_coaxial_collective::kModel,input,persistent,out,next)) { return 0; }
 output[0]=out.upper;output[1]=out.lower;
 for(int i=0;i<3;++i) { output[2+i]=out.servos[i]; }
 for(int i=0;i<5;++i) { persistent[i]=next[i]; }
 return 1;
}
'''
results=[]
with tempfile.TemporaryDirectory(prefix='yaw-balance-') as work:
 work=Path(work);(work/'wrapper.cpp').write_text(wrapper)
 subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC','-I',str(core),str(work/'wrapper.cpp'),'-o',str(work/'core.so')],check=True)
 lib=ctypes.CDLL(str(work/'core.so'));fp=ctypes.POINTER(ctypes.c_float)
 lib.tick.argtypes=[ctypes.c_float,ctypes.c_float,ctypes.c_float,fp,fp];lib.tick.restype=ctypes.c_int
 for rho in [.6,.65,.7,.75,.8]:
  state=np.zeros(5,dtype=np.float32);command=np.zeros(5,dtype=np.float32)
  speed=np.array([963.9750,1041.0930]);servo=-.16;rate=0.;az=-g;records=[]
  for n in range(7500):
   t=n*dt;target=.15 if 4<=t<6 else (-.15 if 8<=t<10 else 0.)
   assert lib.tick(rate,target,az,state.ctypes.data_as(fp),command.ctypes.data_as(fp))
   demand=np.interp(command[:2],table[:,0],table[:,1])
   speed += (demand-speed)*(1-np.exp(-dt/np.array([.04,.05])))
   servo += (float(command[2:].mean())-servo)*(1-np.exp(-dt/.06))
   T=kt[0]*speed[0]**2+kt[1]*speed[1]**2+kt[2]*speed.prod()
   attenuation=1-.195*abs(servo);angle=.245*servo
   # Existing geometry: radius .05 m, total grid thrust share .35.
   grid_yaw=.05*.35*T*attenuation*np.sin(angle)
   vertical_force=T*(.65+.35*attenuation*np.cos(angle));az=-vertical_force/mass
   mz=kq*(speed[0]**2-rho*speed[1]**2)+grid_yaw
   rate+=mz/inertia*dt
   predicted=np.interp(command[:2],table[:,0],table[:,1])
   ratio=max(predicted)/max(min(predicted),1e-9)
   assert ratio<=1.15001 and max(command[:2])<=.900001
   assert abs(float(state[4]))*.0628326535<=.003001
   records.append([t,target,rate,*command[:2]*.5,servo,ratio,vertical_force,mz])
  a=np.array(records);last=a[:,0]>=13
  results.append({'assumed_lower_upper_kq_ratio':rho,'final_mean_rate_rad_s':float(a[last,2].mean()),
   'max_abs_rate_rad_s':float(abs(a[:,2]).max()),'final_mean_physical_commands':a[last,3:5].mean(0).tolist(),
   'max_predicted_speed_ratio':float(a[:,6].max()),'max_physical_motor_command':float(a[:,3:5].max()),
   'final_vertical_force_n':float(a[last,7].mean()),'final_yaw_integral_moment_n_m':float(state[4]*.0628326535)})
  np.savetxt(OUT/f'reduced_model_rho_{rho:.2f}.csv',a,delimiter=',',header='t,target_r,r,upper,lower,common,predicted_max_min_speed_ratio,vertical_force,yaw_moment',comments='')
(OUT/'verification.json').write_text(json.dumps({'model':'reduced nonlinear yaw/vertical model, constant torque coefficients; not flight validation','results':results},indent=2))
print(json.dumps(results,indent=2))
