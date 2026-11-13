"""Reproducible offline diagnosis; no firmware/configuration changes.
Run with /home/ryan/miniconda3/envs/rl-flight/bin/python.
"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/rl-flight-matplotlib')
import csv,json,re
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
z=np.load(OUT/'topics.npz'); meta=json.loads((OUT/'metadata.json').read_text())
def ts(a): return a['timestamp'].astype(float)*1e-6
def interp(key,field,t,hold=False):
 a=z[key];x=ts(a);v=a[field]
 if hold:return v[np.clip(np.searchsorted(x,t,side='right')-1,0,len(x)-1)]
 if v.ndim==1:return np.interp(t,x,v)
 return np.stack([np.interp(t,x,v[:,i]) for i in range(v.shape[1])],axis=-1)
for key in z.files:
 assert np.all(np.diff(z[key]['timestamp'].astype('i8'))>=0),key
assert np.max(abs(np.linalg.norm(z['vehicle_attitude_0']['q'],axis=1)-1))<1e-4
m=z['actuator_motors_0'];t=ts(m);keep=(t>=84)&(t<=119);t=t[keep];mot=m['control'][keep,:2]
rates=interp('vehicle_angular_velocity_0','xyz',t)
rc=np.stack([interp('manual_control_setpoint_0',f,t,hold=True) for f in ['roll','pitch','yaw','throttle']],axis=-1)
q=interp('vehicle_attitude_0','q',t);q/=np.linalg.norm(q,axis=1)[:,None]
rpy=Rotation.from_quat(q[:,[1,2,3,0]]).as_euler('xyz',degrees=True)
servo=interp('actuator_servos_0','control',t)[:,:3]
esc=z['esc_status_0'];rpm=np.stack([np.interp(t,ts(esc),esc['esc']['esc_rpm'][:,i]) for i in range(2)],axis=-1)
yaw_ref=1.2*rc[:,2]
sat=np.isfinite(mot[:,1])&(mot[:,1]>=.49999)
ids=np.where(np.diff(np.r_[False,sat,False]))[0]
intervals=[{'start_s':float(t[s]),'last_sample_s':float(t[e-1]),'duration_s':float(t[e-1]-t[s]+.005)} for s,e in zip(ids[::2],ids[1::2])]
windows=[(88,91),(91,94),(94,96),(96,97),(97,98),(98,98.83),(98.83,100.99),(101,104),(106,110),(110,113),(113,116),(116,117)]
rows=[]
for lo,hi in windows:
 k=(t>=lo)&(t<hi);u=mot[k];rr=rates[k,2];ref=yaw_ref[k];rp=rpm[k]
 rows.append({'start_s':lo,'end_s':hi,'upper_median':float(np.nanmedian(u[:,0])),'lower_median':float(np.nanmedian(u[:,1])),
 'yaw_rate_mean_rad_s':float(rr.mean()),'yaw_reference_mean_rad_s':float(ref.mean()),'yaw_rate_rmse_rad_s':float(np.sqrt(np.mean((rr-ref)**2))),
 'upper_rpm_median':float(np.median(rp[:,0])),'lower_rpm_median':float(np.median(rp[:,1])),
 'apparent_kq_lower_upper_ratio':float(np.median((rp[:,0]/rp[:,1])**2)),
 'saturation_percent':float(sat[k].mean()*100),'servo_saturation_percent':float((np.max(abs(servo[k]),axis=1)>=.9999).mean()*100),
 'yaw_deg_change':float(rpy[k,2][-1]-rpy[k,2][0])})
# Read the generated model used by the current candidate. Treat provenance as
# source-level correspondence, not an in-log checksum assertion.
payload=(ROOT/'px4_trans/px4/src/modules/nn_control/LqiCoaxialCollectivePayload.hpp').read_text()
def values(name):
 s=re.search(r'\b'+name+r'\[\d+\]\s*=\s*\{(.*?)\};',payload,re.S).group(1)
 return np.array([float(x.strip().rstrip('f')) for x in s.split(',') if x.strip()])
grid=values('kMotorEffectivenessPoints');eff=values('kMotorEffectiveness').reshape(-1,4);table=values('kUpperTable').reshape(-1,2)
kr=values('kAttitudeGain').reshape(3,8)[2,4];ki=values('kAttitudeGain').reshape(3,8)[2,7]
base=mot.sum(axis=1) # equals pre-map base only for symmetric, UNSATURATED allocation
b=np.interp(base,grid,eff[:,2]);mz=2*b*(mot[:,0]-mot[:,1])
yaw_integral=(-mz-kr*(rates[:,2]-yaw_ref))/ki
yaw_integral[sat|~np.isfinite(base)]=np.nan
# Frozen measured state and persistent integral. This is a sensitivity, not a
# simulated alternate flight; saturation hides raw lower request.
sensitivity=[]
for scale in [0.6,0.3,0.0]:
 newref=rc[:,2]*scale;dref=newref-yaw_ref
 nominal_yaw_slope=np.interp(.6268292684,grid,eff[:,2])
 dl=-kr*dref/(4*nominal_yaw_slope)
 k=(t>=98.83)&(t<100.99)
 sensitivity.append({'new_full_stick_rate_rad_s':scale,'assumed_nominal_yaw_slope':float(nominal_yaw_slope),'mean_lower_command_change_during_saturation':float(np.nanmean(dl[k])),
 'max_abs_lower_command_change_during_saturation':float(np.nanmax(abs(dl[k])))})
trim=[]
T=.66*9.80665;kt=2.5736e-6;kc=1.2868e-6
for rho in [1,.8,.75,.7,.65,.6,.5]:
 ratio=np.sqrt(1/rho);wu=np.sqrt(T/(kt*(1+ratio**2)+kc*ratio));wl=wu*ratio
 cmd=.5*np.interp([wu,wl],table[:,1],table[:,0])
 trim.append({'kq_lower_upper_ratio':rho,'physical_upper':float(cmd[0]),'physical_lower':float(cmd[1]),'speed_rad_s':[float(wu),float(wl)],'lower_headroom':float(.5-cmd[1])})
# Match external motion capture by independently recorded FC boot timestamps.
ground=Path('/home/ryan/TeleaiTrans_dev/Dock_ground/logs/flights/flight_20261104T114002_007Z_aircraft-01_43')
gr=list(csv.DictReader(open(str(ground)+'.csv')));offsets=[];attitude_matches=[]
for line in open(str(ground)+'.pair.jsonl'):
 x=json.loads(line)
 if x.get('message_id')==19:
  f=x['data']['fields'];bt=f['timestamp_us']*1e-6;offsets.append(x['host_mono_ns']*1e-9-bt)
  if 84<bt<119:
   uq=interp('vehicle_attitude_0','q',np.array([f['attitude_sample_us']*1e-6]))[0]
   attitude_matches.append(float(abs(np.dot(uq,np.array(f['q'])))))
offset=np.median(offsets);gt=np.array([float(r['mono_ns'])*1e-9-offset for r in gr]);gp=np.array([[float(r['position_'+f+'_m']) for f in 'xyz'] for r in gr]);gq=np.array([[float(r['pose_q_'+f]) for f in ['x','y','z','w']] for r in gr]);gyaw=Rotation.from_quat(gq).as_euler('xyz',degrees=True)[:,2]
ground_windows=[]
for lo,hi in [(91,94),(94,97),(98.83,100.99),(106,110),(113,116)]:
 k=(gt>=lo)&(gt<hi);ground_windows.append({'start_s':lo,'end_s':hi,'position_mean_m':gp[k].mean(0).tolist(),'yaw_change_deg':float(gyaw[k][-1]-gyaw[k][0]),'pose_fresh_percent':float(np.mean([gr[i]['pose_fresh']=='1' for i in np.where(k)[0]])*100)})
metrics={'source':meta['source'],'time_basis':'FC boot seconds','motor_output_domain':'actuator_motors after x0.5 map, firmware upper bound 0.5',
 'quality':{'dropout_count':len(meta['dropouts_ms']),'dropout_total_ms':sum(meta['dropouts_ms']),'last_flight_rate_max_gap_s':float(np.diff(ts(z['vehicle_angular_velocity_0']))[(ts(z['vehicle_angular_velocity_0'])[:-1]>84)&(ts(z['vehicle_angular_velocity_0'])[:-1]<119)].max()),'quaternion_and_monotonic_checks_passed':True},
 'saturation_intervals':intervals,'windows':rows,'model_trim_sweep':trim,'frozen_state_yaw_limit_sensitivity':sensitivity,'yaw_gain_rate':float(kr),'yaw_gain_integral':float(ki),
 'ground':{'session':str(ground),'boot_host_offset_spread_ms':float(np.ptp(offsets)*1000),'min_quaternion_abs_dot':min(attitude_matches),'unique_pose_sequences':len(set(r['pose_sequence'] for r in gr)),'max_pose_age_ms':max(float(r['pose_age_ms']) for r in gr),'windows':ground_windows},
 'limitations':['No recorded in-flight model checksum/internal integrator/raw actuator demand','Apparent torque-coefficient ratios assume steady yaw and negligible external torque; not identified coefficients','Counterfactual is frozen-state sensitivity only; saturated raw demand is not recoverable from clipped outputs alone','ESC voltage/current are zero/unavailable; supply sag cannot be verified']}
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2))
with (OUT/'windows.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (OUT/'timeline.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['boot_s','upper_command','lower_command','upper_rpm','lower_rpm','p','q','r','yaw_reference','rc_yaw','rc_throttle','roll_deg','pitch_deg','yaw_deg','servo1','servo2','servo3','inferred_yaw_integral_unsaturated'])
 for i in range(len(t)):w.writerow([t[i],*mot[i],*rpm[i],*rates[i],yaw_ref[i],rc[i,2],rc[i,3],*rpy[i],*servo[i],yaw_integral[i]])
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.25})
fig,axes=plt.subplots(6,1,figsize=(14,14),sharex=True,layout='constrained')
axes[0].plot(t,mot[:,0],label='Upper');axes[0].plot(t,mot[:,1],label='Lower');axes[0].axhline(.5,color='r',ls='--',label='Firmware cap');axes[0].set_ylabel('Motor command');axes[0].set_ylim(0,.53)
axes[1].plot(t,rpm[:,0],label='Upper RPM');axes[1].plot(t,rpm[:,1],label='Lower RPM');axes[1].set_ylabel('Mechanical RPM')
axes[2].plot(t,yaw_ref,label='Target body r (1.2 x stick)');axes[2].plot(t,rates[:,2],label='Measured body r',alpha=.8);axes[2].set_ylabel('Yaw rate (rad/s)')
axes[3].plot(t,rpy[:,2],label='Onboard heading');gk=(gt>=84)&(gt<=119);gref=np.interp(91,gt,gyaw);uref=np.interp(91,t,rpy[:,2]);axes[3].plot(gt[gk],-gyaw[gk]+gref+uref,label='Mocap heading (sign/offset aligned at 91s)',alpha=.8);axes[3].set_ylabel('Heading (deg)')
axes[4].plot(t,rc[:,3],label='Throttle stick');axes[4].plot(t,rc[:,2],label='Yaw stick');axes[4].set_ylabel('RC stick')
axes[5].plot(gt[gk],gp[gk,2],label='Mocap height');axes[5].set_ylabel('Height (m)');axes[5].set_xlabel('FC boot time (s)')
for ax in axes:
 ax.axvspan(98.830611,100.990683,color='r',alpha=.1);ax.set_xlim(84,119);ax.legend(loc='upper left',fontsize=9,ncol=3)
fig.suptitle('Last flight: lower-motor saturation and moving yaw trim\nShading: continuous lower-output saturation, 98.831-100.991 s')
fig.savefig(OUT/'yaw_timeline.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
ax=axes[0]
for lo,hi,col in [(91,94,'#1976d2'),(94,97,'#ff9800'),(98.83,100.99,'#d32f2f'),(106,110,'#388e3c'),(113,116,'#7b1fa2')]:
 k=(t>=lo)&(t<hi);ax.scatter(mot[k,0][::5],mot[k,1][::5],s=7,alpha=.5,color=col,label=f'{lo}-{hi} s')
ax.plot([0,.5],[0,.5],'k--',label='Equal-command model trim');ax.axhline(.5,color='r',ls=':');ax.set(xlim=(0,.5),ylim=(0,.53),xlabel='Upper physical command',ylabel='Lower physical command');ax.legend(fontsize=8);ax.set_title('Observed outputs (saturated points are NOT trim)')
ax=axes[1];rr=np.array([r['kq_lower_upper_ratio'] for r in trim]);ax.plot(rr,[r['physical_upper'] for r in trim],'o-',label='Upper');ax.plot(rr,[r['physical_lower'] for r in trim],'o-',label='Lower');ax.axhline(.5,color='r',ls='--',label='Firmware cap');ax.axvspan(.6,.8,color='g',alpha=.1);ax.set(xlabel='Assumed lower/upper torque coefficient',ylabel='Predicted physical hover command',title='Model trim at 0.66 kg; same existing thrust/speed model');ax.legend()
fig.savefig(OUT/'trim_analysis.png',dpi=150);plt.close(fig)
print(json.dumps({'windows':rows,'trim':trim,'sensitivity':sensitivity,'ground':metrics['ground'],'quality':metrics['quality']},indent=2))
