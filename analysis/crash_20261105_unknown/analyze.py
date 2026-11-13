from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/rl-flight-matplotlib')
import numpy as np,json,csv
from scipy.spatial.transform import Rotation
from scipy.signal import butter,sosfiltfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
z=np.load(OUT/'topics.npz');meta=json.loads((OUT/'metadata.json').read_text())
for key in z.files:assert np.all(np.diff(z[key]['timestamp'].astype('i8'))>=0),key
assert np.max(abs(np.linalg.norm(z['vehicle_attitude_0']['q'],axis=1)-1))<1e-4
def ts(a):return a['timestamp']*1e-6
def interp(key,field,t,hold=False):
 a=z[key];x=ts(a);v=a[field]
 if hold:return v[np.clip(np.searchsorted(x,t,side='right')-1,0,len(x)-1)]
 if v.ndim==1:return np.interp(t,x,v)
 return np.stack([np.interp(t,x,v[:,i]) for i in range(v.shape[1])],axis=-1)
a=z['actuator_servos_0'];t=ts(a);k=(t>=14)&(t<=29);t=t[k];servo=a['control'][k,:3];mot=interp('actuator_motors_0','control',t)[:,:2]
q=interp('vehicle_attitude_0','q',t);q/=np.linalg.norm(q,axis=1)[:,None];rpy=Rotation.from_quat(q[:,[1,2,3,0]]).as_euler('xyz',degrees=True);rates=interp('vehicle_angular_velocity_0','xyz',t)
rc=np.stack([interp('manual_control_setpoint_0',f,t,hold=True) for f in ['roll','pitch','yaw','throttle']],axis=-1)
esc=z['esc_status_0'];rpm=np.stack([np.interp(t,ts(esc),esc['esc']['esc_rpm'][:,i]) for i in range(2)],axis=-1)
tab=np.array([[0,0],[.08,0],[.1,446.991],[.2,578.304],[.3,717.411],[.4,831.996],[.5,936.197],[.6,988.958],[.8,1093.479],[1,1200]])
predicted=np.stack([np.interp(2*mot[:,i],tab[:,0],tab[:,1]) for i in range(2)],axis=-1);pred_ratio=predicted[:,0]/np.maximum(predicted[:,1],1)
common=servo.mean(1)
# Exact local inversion only when common is NOT endpoint-retreated or ramp-capped.
# Below hover: common=-.16*T_desired/(mg), servo_scale=T_desired/(mg).
scale=-common/.16;mx=(servo[:,0]-common)*scale/72.0721436;my=(servo[:,2]-servo[:,1])*scale/124.832611
proxy_valid=(common<-.01)&(common>-.159)&(np.max(abs(servo),axis=1)<.9999)
mx[~proxy_valid]=np.nan;my[~proxy_valid]=np.nan
base=Path('/home/ryan/TeleaiTrans_dev/Dock_ground/logs/flights/flight_20261105T030533_568Z_aircraft-01_2')
ground=list(csv.DictReader(open(str(base)+'.csv')));diag={i:[] for i in [16,19]};offset=[];qmatches=[]
for line in open(str(base)+'.pair.jsonl'):
 x=json.loads(line)
 if x.get('message_id') in diag:diag[x['message_id']].append(x['data']['fields'])
 if x.get('message_id')==19:
  f=x['data']['fields'];offset.append(x['host_mono_ns']*1e-9-f['timestamp_us']*1e-6)
  if 18<f['timestamp_us']*1e-6<27:
   iq=interp('vehicle_attitude_0','q',np.array([f['attitude_sample_us']*1e-6]))[0];qmatches.append(float(abs(iq@np.array(f['q']))))
gt=np.array([float(r['mono_ns'])*1e-9-np.median(offset) for r in ground]);gp=np.array([[float(r['position_'+f+'_m']) for f in 'xyz'] for r in ground]);gq=np.array([[float(r['pose_q_'+f]) for f in ['x','y','z','w']] for r in ground]);ge=Rotation.from_quat(gq).as_euler('xyz',degrees=True);fresh=np.array([r['pose_fresh']=='1' for r in ground]);replay=np.load(OUT/'replay.npz');rt=replay['t'];ro=replay['outputs']
metrics={'source':meta['source'],'time_basis':'FC boot seconds','quality':{'dropout_count':len(meta['dropouts_ms']),'dropout_total_ms':sum(meta['dropouts_ms']),'checks':'topic timestamps monotonic; quaternion norm passed'},'ground':{'session':str(base),'min_attitude_abs_dot':min(qmatches),'offset_spread_ms':float(np.ptp(offset)*1000),'fresh_percent_after_18s':float(fresh[(gt>=18)&(gt<29)].mean()*100)},'windows':[]}
for lo,hi in [(18,22),(23,24),(24,25),(25,26),(26,27),(27,27.5)]:
 k=(t>=lo)&(t<hi);gk=(gt>=lo)&(gt<hi)&fresh;v=k&proxy_valid
 d19=[x for x in diag[19] if lo<=x['timestamp_us']*1e-6<hi]
 metrics['windows'].append({'start_s':lo,'end_s':hi,'rpy_min_deg':rpy[k].min(0).tolist(),'rpy_max_deg':rpy[k].max(0).tolist(),'ground_roll_range_deg':ge[gk,0][[np.argmin(ge[gk,0]),np.argmax(ge[gk,0])]].tolist(),'ground_height_mean_m':float(gp[gk,2].mean()),'motor_median':np.nanmedian(mot[k],axis=0).tolist(),'esc_rpm_median':np.median(rpm[k],axis=0).tolist(),'predicted_upper_lower_speed_ratio_median':float(np.median(pred_ratio[k])),'upper_speed_ratio_boundary_sample_percent':float(100*np.mean(pred_ratio[k]>1.1495)),'actual_upper_lower_rpm_ratio_median':float(np.median(rpm[k,0]/np.maximum(rpm[k,1],1))),'grid_common_median':float(np.nanmedian(common[k])),'servo_endpoint_percent':(100*np.nanmean(abs(servo[k])>.999,axis=0)).tolist(),'inferred_roll_moment_lower_limit_sample_percent':float(100*np.mean(mx[v]<-.00895)),'moment_proxy_valid_samples':int(v.sum()),'roll_moment_proxy_median':float(np.nanmedian(mx[k])),'rc_yaw_median':float(np.median(rc[k,2])),'actual_yaw_rate_mean_rad_s':float(rates[k,2].mean()),'motor_output_max_age_ms_from_ground':max((x['timestamp_us']-x['motors_timestamp_us'])/1000 for x in d19)})
# Native record change times, without interpolating across gaps.
a=z['vehicle_angular_velocity_0'];mask=(ts(a)>26)&(abs(a['xyz'][:,2])>1);metrics['yaw_exceeds_1_rad_s_at']=float(ts(a)[mask][0])
a=z['manual_control_setpoint_0'];mask=(ts(a)>26)&(a['throttle']<-.8);metrics['throttle_below_minus_0p8_at']=float(ts(a)[mask][0])
a=z['actuator_motors_0'];mask=(ts(a)>26)&np.isfinite(a['control'][:,0])&(a['control'][:,0]<=.001);metrics['motors_zero_at']=float(ts(a)[mask][0])
metrics['stop_guard_log_s']=28.118694
# Cross-IMU comparison below 8 Hz: confirms large motion is not single-gyro noise.
tt=np.arange(23,27.5,.002);gg=[]
for key in ['sensor_gyro_0','sensor_gyro_1']:
 a=z[key];v=np.stack([np.interp(tt,ts(a),a[f]) for f in 'xyz'],axis=-1);gg.append(sosfiltfilt(butter(3,8,fs=500,output='sos'),v,axis=0))
metrics['gyro_lowpass_correlations_xyz']=[float(np.corrcoef(gg[0][:,i],gg[1][:,i])[0,1]) for i in range(3)]
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2))
with (OUT/'timeline.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['boot_s','roll','pitch','yaw','p','q','r','upper','lower','upper_rpm','lower_rpm','servo1','servo2','servo3','common','pred_upper_lower_ratio','inferred_Mx','inferred_My','rc_roll','rc_pitch','rc_yaw','rc_throttle'])
 for i in range(len(t)):w.writerow([t[i],*rpy[i],*rates[i],*mot[i],*rpm[i],*servo[i],common[i],pred_ratio[i],mx[i],my[i],*rc[i]])
plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.25})
fig,axes=plt.subplots(7,1,figsize=(14,17),sharex=True,layout='constrained')
axes[0].plot(t,rpy[:,0],label='Onboard roll');gk=fresh&(gt>=17.6)&(gt<=29);axes[0].plot(gt[gk],ge[gk,0],label='Mocap roll (mounting offset not removed)',alpha=.8);axes[0].plot(t,rc[:,0]*10,ls='--',label='Roll stick x 10 deg');axes[0].set_ylabel('Roll (deg)');axes[0].set_ylim(-35,25)
axes[1].plot(t,mx*1000,label='Roll request inferred from allocation (valid only)');axes[1].axhline(-9,ls='--',color='r',label='Roll moment cap');axes[1].set_ylabel('Model Mx (mN m)');axes[1].set_ylim(-11,11)
axes[2].plot(t,mot[:,0],label='Upper');axes[2].plot(t,mot[:,1],label='Lower');axes[2].axhline(.45,color='r',ls='--',label='Output cap');axes[2].set_ylabel('Motor command');axes[2].set_ylim(-.02,.48)
axes[3].plot(t,pred_ratio,label='Predicted upper/lower speed');axes[3].plot(t,rpm[:,0]/np.maximum(rpm[:,1],1),label='Measured upper/lower RPM',alpha=.8);axes[3].axhline(1.15,ls='--',color='r',label='Predicted speed boundary');axes[3].set_ylim(.7,1.6);axes[3].set_ylabel('Speed ratio')
axes[4].plot(t,rates[:,2],label='Actual body r');axes[4].plot(rt,ro[:,10],label='Approx. filtered target r');axes[4].set_ylabel('Yaw rate (rad/s)');axes[4].set_ylim(-3.5,.5)
axes[5].plot(t,servo[:,0],label='Servo 1');axes[5].plot(t,servo[:,1],label='Servo 2');axes[5].plot(t,servo[:,2],label='Servo 3');axes[5].plot(t,common,'k',lw=1.5,label='Common');axes[5].set_ylabel('Servo command');axes[5].set_ylim(-1.1,1.1)
axes[6].plot(gt[gk],gp[gk,2],label='Mocap height (m)');axes[6].plot(t,rc[:,3],label='Throttle stick');axes[6].set_ylabel('Height / stick');axes[6].set_xlabel('FC boot seconds')
for ax in axes:
 ax.axvspan(24,26.9,color='orange',alpha=.10);ax.axvspan(26.9,27.616,color='red',alpha=.10);ax.axvline(27.616,color='gray',ls=':');ax.axvline(28.1187,color='black',ls=':');ax.set_xlim(17.5,28.5);ax.legend(loc='upper left',fontsize=8,ncol=3)
fig.suptitle('2026-11-05: roll oscillation first, yaw allocation boundary later\nOrange: growing roll oscillation; red: yaw spin before zero throttle; dotted: motors zero / 30-deg stop')
fig.savefig(OUT/'crash_timeline.png',dpi=150);plt.close(fig)
