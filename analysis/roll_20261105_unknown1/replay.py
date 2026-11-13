"""Approximate controller-state reconstruction, not a counterfactual flight.
ULog gaps and unlogged scheduler timing prevent exact replay.
"""
from pathlib import Path
import ctypes,tempfile,subprocess,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
z=np.load(OUT/'topics.npz');t=np.arange(9.86989,35.510,.002)
def ip(key,field,hold=False):
 a=z[key];tt=a['timestamp']*1e-6;v=a[field]
 if hold:return v[np.clip(np.searchsorted(tt,t,side='right')-1,0,len(tt)-1)]
 return np.stack([np.interp(t,tt,v[:,i]) for i in range(v.shape[1])],axis=-1)
q=ip('vehicle_attitude_0','q');q/=np.linalg.norm(q,axis=1)[:,None];rate=ip('vehicle_angular_velocity_0','xyz');acc=ip('vehicle_acceleration_0','xyz');rc=np.stack([ip('manual_control_setpoint_0',f,True) for f in ['roll','pitch','yaw','throttle']],axis=1);actual=np.c_[ip('actuator_motors_0','control')[:,:2],ip('actuator_servos_0','control')[:,:3]]
inputs=np.c_[q,rate,acc,rc].astype('f4');outputs=np.zeros((len(t),13),dtype='f4')
source=r'''
#include "LqiCoaxialCollectiveCore.hpp"
#include "LqiCoaxialCollectivePayload.hpp"
#include "LqiManualReference.hpp"
static LqiCoaxialCollectiveCore core{lqi_coaxial_collective::kModel};
static LqiCoaxialManualReference reference{lqi_coaxial_collective::kModel};
extern "C" void tick(const float *x, float *out) {
 LqiCoaxialCollectiveCore::Input in{};LqiCoaxialCollectiveCore::ControlOutput y{};
 reference.build(x,x+4,x[10],x[11],x[12],x[13],in);
 for(int i=0;i<3;++i) { in.acceleration_b[i]=x[7+i]; }
 core.step(in,y);out[0]=y.upper*.5f;out[1]=y.lower*.5f;
 for(int i=0;i<3;++i) { out[2+i]=y.servos[i]; }
 for(int i=0;i<5;++i) { out[5+i]=core.persistent()[i]; }
 out[10]=in.target_angular_velocity_b[2];
 out[11]=std::atan2(2*(in.target_attitude_q_wb[0]*in.target_attitude_q_wb[3]+in.target_attitude_q_wb[1]*in.target_attitude_q_wb[2]),1-2*(in.target_attitude_q_wb[2]*in.target_attitude_q_wb[2]+in.target_attitude_q_wb[3]*in.target_attitude_q_wb[3]));
 out[12]=in.collective_base;
}
'''
with tempfile.TemporaryDirectory(prefix='crash-replay-') as work:
 p=Path(work);(p/'replay.cpp').write_text(source);subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC','-I',str(OUT/'controller_v2_source'),str(p/'replay.cpp'),'-o',str(p/'replay.so')],check=True)
 lib=ctypes.CDLL(str(p/'replay.so'));fp=ctypes.POINTER(ctypes.c_float);lib.tick.argtypes=[fp,fp]
 for i in range(len(t)):lib.tick(inputs[i].ctypes.data_as(fp),outputs[i].ctypes.data_as(fp))
np.savez_compressed(OUT/'replay.npz',t=t,inputs=inputs,outputs=outputs,actual=actual)
metrics=[]
for lo,hi in [(13,16),(19,22),(27,30),(30,32),(32,35)]:
 k=(t>=lo)&(t<hi);err=outputs[k,:5]-actual[k];metrics.append({'start':lo,'end':hi,'output_rmse':np.sqrt(np.nanmean(err*err,axis=0)).tolist(),'integrals_mean':outputs[k,7:10].mean(0).tolist(),'roll_integral_moment_mean':float(-.0997526571*outputs[k,7].mean()),'target_yaw_deg_mean':float(outputs[k,11].mean()*180/np.pi),'target_rate_mean':float(outputs[k,10].mean())})
(OUT/'replay_metrics.json').write_text(json.dumps({'limitation':'Approximate reconstructed states, not logged values; missing samples and loop schedule prevent exact equivalence','windows':metrics},indent=2));print(json.dumps(metrics,indent=2))
