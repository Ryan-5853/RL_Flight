from pathlib import Path
import os,json,csv,collections
os.environ['MPLCONFIGDIR']='/tmp/rl-flight-matplotlib'
import numpy as np
from scipy.signal import welch
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent
base=Path('/home/ryan/TeleaiTrans_dev/Dock_ground/logs/flights/flight_20261105T071243_578Z_aircraft-01_1')
z=np.load(O/'topics.npz');m=json.loads((O/'metadata.json').read_text())
def ip(z,key,field,t,hold=False):
 a=z[key];tt=a['timestamp']*1e-6;v=a[field]
 if hold:return v[np.clip(np.searchsorted(tt,t,side='right')-1,0,len(tt)-1)]
 if v.ndim==1:return np.interp(t,tt,v)
 return np.stack([np.interp(t,tt,v[:,j]) for j in range(v.shape[1])],1)
off=[];dots=[];traces=[]
for l in open(str(base)+'.pair.jsonl'):
 x=json.loads(l);f=x.get('data',{}).get('fields',{})
 if x.get('message_id')==19:
  off.append(x['host_mono_ns']*1e-9-f['timestamp_us']*1e-6)
  if 12e6<f['attitude_sample_us']<84e6:
   q=ip(z,'vehicle_attitude_0','q',[f['attitude_sample_us']*1e-6])[0];dots.append(abs(q@np.array(f['q'])))
 if x.get('message_id')==18:traces.append(f)
g=list(csv.DictReader(open(str(base)+'.csv')));gt=np.array([float(r['mono_ns'])*1e-9-np.median(off) for r in g]);ge=R.from_quat([[float(r['pose_q_'+f]) for f in ['x','y','z','w']] for r in g]).as_euler('xyz',degrees=True);pos=np.array([[float(r['position_'+f+'_m']) for f in ['x','y','z']] for r in g])
res={'ground_session':str(base),'alignment_q_dot_min':float(min(dots)),'offset_p5_p50_p95_s':np.percentile(off,[5,50,95]).tolist(),'ground_fresh_percent':100*np.mean([r['pose_fresh']=='1' for r in g]),'pose_age_p99_max_ms':np.percentile([float(r['pose_age_ms']) for r in g],[99,100]).tolist(),'dropout_total_ms':sum(m['dropouts_ms']),'trace_count':len(traces),'trace_nonzero_timestamp':sum(f['timestamp_us']>0 for f in traces),'windows':[]}
t=np.arange(10,85,.005)
s=ip(z,'actuator_servos_0','control',t)[:,:3];mot=ip(z,'actuator_motors_0','control',t)[:,:2];rates=ip(z,'vehicle_angular_velocity_0','xyz',t);rc=np.stack([ip(z,'manual_control_setpoint_0',f,t,True) for f in ['roll','pitch','yaw','throttle']],1);q=ip(z,'vehicle_attitude_0','q',t);q/=np.linalg.norm(q,axis=1)[:,None];rot=R.from_quat(q[:,[1,2,3,0]]);e=rot.as_euler('xyz',degrees=True);target=R.from_euler('xyz',np.c_[rc[:,0]*10,-rc[:,1]*10,e[:,2]],degrees=True);err=(target.inv()*rot).as_rotvec()
common=s.mean(1);table=np.array([[0,0],[.08,0],[.1,446.991],[.2,578.304],[.3,717.411],[.4,831.996],[.5,936.197],[.6,988.958],[.8,1093.479],[1,1200]])
w=np.stack([np.interp(2*mot[:,j],table[:,0],table[:,1]) for j in range(2)],1);ratio=w[:,0]/np.maximum(w[:,1],1)
valid=(common>-.0795)&(common<-.01)&(ratio<1.148)&(ratio>1/1.148)&(abs(s).max(1)<.999)
mx=(s[:,0]-common)*(-common/.08)/72.0721436;mx[~valid]=np.nan
# Native samples used for boundary statistics to avoid interpolation smoothing.
a=z['actuator_servos_0'];nt=a['timestamp']*1e-6;ns=a['control'][:,:3];nc=ns.mean(1);nm=ip(z,'actuator_motors_0','control',nt)[:,:2];nw=np.stack([np.interp(2*nm[:,j],table[:,0],table[:,1]) for j in range(2)],1);nr=nw[:,0]/np.maximum(nw[:,1],1);nv=(nc>-.0795)&(nc<-.01)&(nr<1.148)&(nr>1/1.148)&(abs(ns).max(1)<.999);nmx=(ns[:,0]-nc)*(-nc/.08)/72.0721436
for lo,hi in [(12,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,84)]:
 k=(t>=lo)&(t<hi);kg=(gt>=lo)&(gt<hi);v=(nt>=lo)&(nt<hi)&nv
 P=-.0814316645*err[k,0];D=-.0168643519*rates[k,0]
 integ=mx[k]+.0814316645*err[k,0]+.0168643519*rates[k,0];integ[abs(mx[k])>.0088]=np.nan
 res['windows'].append(dict(start=lo,end=hi,target_rp_deg_mean=np.mean(np.c_[rc[k,0]*10,-rc[k,1]*10],0).tolist(),onboard_rp_deg_mean=e[k,:2].mean(0).tolist(),mocap_rp_deg_mean=ge[kg,:2].mean(0).tolist(),stick_abs_above_0p9_pct=(100*np.mean(abs(rc[k,:3])>.9,0)).tolist(),rate_std=rates[k].std(0).tolist(),servo_min_median_max=np.percentile(s[k],[0,50,100],axis=0).tolist(),motor_median=np.median(mot[k],axis=0).tolist(),motor_above_0p445_pct=(100*np.mean(mot[k]>.445,0)).tolist(),native_valid_count=int(v.sum()),roll_negative_cap_pct=float(100*np.mean(nmx[v]<-.00895)),roll_positive_cap_pct=float(100*np.mean(nmx[v]>.00895)),roll_moment_mean=float(np.mean(nmx[v])),inferred_I_median=float(np.nanmedian(integ)),P_mean=float(P.mean()),D_mean=float(D.mean()),D_std=float(D.std()),height_median=float(np.median(pos[kg,2])),position_start_end=pos[kg][[0,-1]].tolist()))
# Identical band integration and sampling for both flights. Equal window length.
res['spectra']=[]
for label,zz,lo,hi in [('previous',np.load(O.parent/'roll_20261105_unknown1/topics.npz'),19,32),('current_early',z,19,32),('current_middle',z,40,53),('current_late',z,67,80)]:
 tt=np.arange(lo,hi,.002);rr=ip(zz,'vehicle_angular_velocity_0','xyz',tt);ss=ip(zz,'actuator_servos_0','control',tt)[:,:3];f,p=welch(rr,500,nperseg=2048,axis=0);_,ps=welch(ss,500,nperseg=2048,axis=0);b=(f>=4)&(f<=8);b2=(f>=1)&(f<=20)
 a=zz['sensor_gyro_fifo_0'];ok=np.arange(32)[None,:]<a['samples'][:,None];ft=(a['timestamp_sample'][:,None]+np.arange(32)[None,:]*a['dt'][:,None])*1e-6;fv=a['x']*a['scale'][:,None];ft=ft[ok];fv=fv[ok];fk=(ft>=lo)&(ft<hi);sp=np.diff(ft[fk]);ut=np.arange(lo,hi,.0005);ff,pp=welch(np.interp(ut,ft,fv),2000,nperseg=8192);bb=(ff>=250)&(ff<=350)
 res['spectra'].append(dict(label=label,window=[lo,hi],rate_4_8_rms=np.sqrt(np.trapezoid(p[b],f[b],axis=0)).tolist(),servo_4_8_rms=np.sqrt(np.trapezoid(ps[b],f[b],axis=0)).tolist(),rate_peak_1_20_hz=f[b2][np.argmax(p[b2],axis=0)].tolist(),raw_x_250_350_rms=float(np.sqrt(np.trapezoid(pp[bb],ff[bb]))),raw_x_peak_hz=float(ff[np.argmax(pp)]),raw_fifo_max_gap_s=float(sp.max()),raw_fifo_gap_above_1ms_count=int(sum(sp>.001))))
(O/'metrics.json').write_text(json.dumps(res,indent=2))
with (O/'timeline.csv').open('w') as out:
 wr=csv.writer(out);wr.writerow(['boot_s','roll','pitch','yaw','stick_roll','stick_pitch','stick_yaw','throttle','motor1','motor2','servo1','servo2','servo3','inferred_Mx'])
 for i in range(len(t)):wr.writerow([t[i],*e[i],*rc[i],*mot[i],*s[i],mx[i]])
plt.rcParams.update({'axes.grid':True,'grid.alpha':.25})
fig,ax=plt.subplots(6,1,figsize=(15,15),sharex=True,layout='constrained')
for j in range(2):
 ax[j].plot(t,rc[:,j]*(10 if j==0 else -10),'k--',label='Stick target');ax[j].plot(t,e[:,j],label='Onboard');ax[j].plot(gt,ge[:,j],alpha=.7,label='Mocap (extrinsics uncalibrated)');ax[j].set_ylabel(['Roll deg','Pitch deg'][j])
ax[2].plot(t,mx*1000,label='Inferred Mx (restricted samples)');ax[2].axhline(-9,color='r',ls='--');ax[2].set_ylabel('Model mN m')
for j in range(3):ax[3].plot(t,s[:,j],label=f'Servo {j+1}')
for j in range(2):ax[4].plot(t,mot[:,j],label=f'Motor {j+1}')
for j in range(3):ax[5].plot(gt,pos[:,j],label=f'Position {"xyz"[j]} m')
for a in ax:a.set_xlim(10,85);a.legend(fontsize=8,loc='upper left',ncol=3)
ax[-1].set_xlabel('FC boot seconds');fig.savefig(O/'tracking.png',dpi=140)
print(json.dumps(res,indent=2))
