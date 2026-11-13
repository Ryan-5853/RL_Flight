from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/rl-flight-matplotlib')
import csv,json,numpy as np
from scipy.spatial.transform import Rotation
from scipy.signal import welch, coherence
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
z=np.load(OUT/'topics.npz');meta=json.loads((OUT/'metadata.json').read_text())
for key in z.files:assert np.all(np.diff(z[key]['timestamp'].astype('i8'))>=0)
def ts(a):return a['timestamp']*1e-6
def ip(key,field,t,hold=False):
 a=z[key];v=a[field];x=ts(a)
 if hold:return v[np.clip(np.searchsorted(x,t,side='right')-1,0,len(x)-1)]
 if v.ndim==1:return np.interp(t,x,v)
 return np.stack([np.interp(t,x,v[:,i]) for i in range(v.shape[1])],axis=-1)
a=z['actuator_servos_0'];keep=(ts(a)>=9)&(ts(a)<=36);t=ts(a)[keep];s=a['control'][keep,:3];common=s.mean(1);mot=ip('actuator_motors_0','control',t)[:,:2];rate=ip('vehicle_angular_velocity_0','xyz',t);rc=np.stack([ip('manual_control_setpoint_0',f,t,True) for f in ['roll','pitch','yaw','throttle']],axis=1)
q=ip('vehicle_attitude_0','q',t);q/=np.linalg.norm(q,axis=1)[:,None];rot=Rotation.from_quat(q[:,[1,2,3,0]]);e=rot.as_euler('xyz');target=Rotation.from_euler('xyz',np.stack([rc[:,0]*np.pi/18,-rc[:,1]*np.pi/18,e[:,2]],axis=1));error=(target.inv()*rot).as_rotvec();e=np.rad2deg(e)
table=np.array([[0,0],[.08,0],[.1,446.991],[.2,578.304],[.3,717.411],[.4,831.996],[.5,936.197],[.6,988.958],[.8,1093.479],[1,1200]])
w=np.stack([np.interp(2*mot[:,i],table[:,0],table[:,1]) for i in range(2)],axis=1);ratio=w[:,0]/np.maximum(w[:,1],1)
# When shared common is unused and low-thrust ramp isn't capped, infer scale
# from common = -.08 * T_desired/(mg). Restrict to native valid samples.
valid=(common>-.0795)&(common<-.01)&(ratio<1.148)&(ratio>1/1.148)&(np.max(abs(s),axis=1)<.999)
mx=(s[:,0]-common)*(-common/.08)/72.0721436;mx[~valid]=np.nan
inferred_integral=mx+.0814316645*error[:,0]+.0168643519*rate[:,0];inferred_integral[abs(mx)>.0088]=np.nan
base=Path('/home/ryan/TeleaiTrans_dev/Dock_ground/logs/flights/flight_20261105T061135_138Z_aircraft-01_3');gr=list(csv.DictReader(open(str(base)+'.csv')));offset=[];dot=[]
for l in open(str(base)+'.pair.jsonl'):
 x=json.loads(l)
 if x.get('message_id')==19:
  f=x['data']['fields'];offset.append(x['host_mono_ns']*1e-9-f['timestamp_us']*1e-6)
  if 13<f['timestamp_us']*1e-6<35:
   qq=ip('vehicle_attitude_0','q',np.array([f['attitude_sample_us']*1e-6]))[0];dot.append(float(abs(qq@np.array(f['q']))))
gt=np.array([float(r['mono_ns'])*1e-9-np.median(offset) for r in gr]);gq=np.array([[float(r['pose_q_'+f]) for f in ['x','y','z','w']] for r in gr]);ge=Rotation.from_quat(gq).as_euler('xyz',degrees=True);height=np.array([float(r['position_z_m']) for r in gr])
metrics={'source':meta['source'],'time_basis':'FC boot seconds','dropouts_ms':meta['dropouts_ms'],'ground':{'session':str(base),'min_attitude_dot':min(dot),'fresh_percent':100*np.mean([r['pose_fresh']=='1' for r in gr]),'max_pose_age_ms':max(float(r['pose_age_ms']) for r in gr)},'windows':[],'pwm':[]}
for lo,hi in [(13,16),(19,22),(27,30),(30,32),(32,33),(33,35)]:
 k=(t>=lo)&(t<hi);gk=(gt>=lo)&(gt<hi);v=k&valid
 metrics['windows'].append({'start':lo,'end':hi,'roll_stick_median':float(np.median(rc[k,0])),'roll_target_deg_mean':float((rc[k,0]*10).mean()),'onboard_roll_deg_mean':float(e[k,0].mean()),'mocap_roll_deg_mean':float(ge[gk,0].mean()),'mocap_roll_range_deg':[float(ge[gk,0].min()),float(ge[gk,0].max())],'height_mean_m':float(height[gk].mean()),'motor_median':np.nanmedian(mot[k],axis=0).tolist(),'servo1_range':np.nanpercentile(s[k,0],[0,50,100]).tolist(),'common_median':float(np.nanmedian(common[k])),'valid_moment_samples':int(v.sum()),'roll_moment_model_median_n_m':float(np.nanmedian(mx[k])),'roll_moment_lower_clip_percent_valid_samples':float(100*np.mean(mx[v]<-.00895)),'inferred_roll_integral_moment_median_n_m':float(np.nanmedian(inferred_integral[k]))})
for channel in [6,7,8]:
 p=meta['params'];low=p[f'PWM_MAIN_MIN{channel}'];high=p[f'PWM_MAIN_MAX{channel}'];off=p[f'PWM_MAIN_DIS{channel}'];metrics['pwm'].append({'channel':channel,'function':p[f'PWM_MAIN_FUNC{channel}'],'minimum':low,'maximum':high,'disarmed':off,'armed_zero':(low+high)/2})
k=(t>=30)&(t<32)
P=-.0814316645*error[k,0];D=-.0168643519*rate[k,0];raw=P+D-.0045
metrics['roll_decomposition_30_32s']={'scope':'Model-domain approximation assuming roll I moment at its observed/inferred -0.0045 cap; not internal trace',
 'P_mean_n_m':float(P.mean()),'D_mean_n_m':float(D.mean()),'D_std_n_m':float(D.std()),'I_assumed_n_m':-.0045,
 'raw_mean_n_m':float(raw.mean()),'clipped_mean_n_m':float(np.clip(raw,-.009,.009).mean()),'output_inferred_mean_n_m':float(np.nanmean(mx[k]))}
metrics['pwm_interpretation']='User confirmed endpoint midpoint is intended zero; disarm values do not participate in flight. No disarm-as-neutral fault inferred.'
# Primary FIFO has two 500-us samples/batch. Verify no holes in this window.
a=z['sensor_gyro_fifo_0'];ok=np.arange(32)[None,:]<a['samples'][:,None];ft=(a['timestamp_sample'][:,None]+np.arange(32)[None,:]*a['dt'][:,None])*1e-6;fv=a['x']*a['scale'][:,None];ft=ft[ok];fv=fv[ok];k=(ft>=19)&(ft<32);spacing=np.diff(ft[k]);uniform=np.arange(19,32,.0005);raw=np.interp(uniform,ft,fv);freq,praw=welch(raw,fs=2000,nperseg=8192)
ct=np.arange(19,32,.002);p=ip('vehicle_angular_velocity_0','xyz',ct)[:,0];sv=ip('actuator_servos_0','control',ct)[:,0];fctrl,pc=welch(p,fs=500,nperseg=2048);_,ps=welch(sv,fs=500,nperseg=2048);peak=freq[np.argmax(praw)];band=(fctrl>=1)&(fctrl<20);control_peak=fctrl[band][np.argmax(pc[band])]
grid=np.arange(19,32,.01);rollg=np.interp(grid,gt,ge[:,0]);ground_rate=np.gradient(np.deg2rad(rollg),.01);fg,pg=welch(ground_rate,fs=100,nperseg=512);b=(fg>1)&(fg<20);gp=fg[b][np.argmax(pg[b])];fc,coh=coherence(ip('vehicle_angular_velocity_0','xyz',grid)[:,0],ground_rate,fs=100,nperseg=256)
metrics['spectrum']={'primary_fifo_max_spacing_s':float(spacing.max()),'primary_fifo_peak_hz':float(peak),'control_roll_rate_peak_hz':float(control_peak),'mocap_approx_roll_rate_peak_1_20_hz':float(gp),'gyro_mocap_roll_rate_coherence_near_5p5_hz':float(coh[np.argmin(abs(fc-5.5))]),'control_roll_rate_std_rad_s':float(p.std()),'control_roll_rate_abs_above_0p3_fraction':float(np.mean(abs(p)>.3)),'note':'Native primary FIFO nearly continuous at 2kHz; 500Hz logged raw gyro has a ~198Hz alias and should not be used to identify the true main frequency.'}
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2))
with (OUT/'timeline.csv').open('w') as stream:
 writer=csv.writer(stream);writer.writerow(['boot_s','roll_deg','pitch_deg','yaw_deg','p','q','r','rc_roll','rc_pitch','rc_yaw','rc_throttle','upper','lower','servo1','servo2','servo3','common','inferred_Mx','inferred_I_roll_moment'])
 for i in range(len(t)):writer.writerow([t[i],*e[i],*rate[i],*rc[i],*mot[i],*s[i],common[i],mx[i],inferred_integral[i]])
plt.rcParams.update({'axes.grid':True,'grid.alpha':.25,'font.size':9})
fig,axes=plt.subplots(5,1,figsize=(14,12),sharex=True,layout='constrained')
axes[0].plot(t,rc[:,0]*10,'k--',label='Roll target from stick');axes[0].plot(t,e[:,0],label='Onboard roll');axes[0].plot(gt,ge[:,0],label='Mocap roll (mounting not calibrated)',alpha=.8);axes[0].set_ylim(-13,12);axes[0].set_ylabel('Roll (deg)')
axes[1].plot(t,1000*mx,label='Inferred model roll moment (restricted samples)');axes[1].axhline(-9,color='r',ls='--',label='Moment request cap');axes[1].set_ylabel('Model Mx (mN m)')
axes[2].plot(t,s[:,0],label='Servo 1');axes[2].plot(t,s[:,1],label='Servo 2');axes[2].plot(t,s[:,2],label='Servo 3');axes[2].plot(t,common,'k',label='Common');axes[2].set_ylabel('Servo command')
axes[3].plot(t,mot[:,0],label='Upper');axes[3].plot(t,mot[:,1],label='Lower');axes[3].set_ylabel('Motor command')
axes[4].plot(gt,height,label='Mocap height (m)');axes[4].plot(t,rc[:,3],label='Throttle stick');axes[4].set_xlabel('FC boot time (s)')
for ax in axes:
 ax.set_xlim(11,35.6);ax.axvline(35.351665,color='gray',ls=':');ax.axvspan(30,32,color='orange',alpha=.1);ax.legend(loc='upper left',fontsize=8,ncol=3)
fig.suptitle('v2 flight: persistent negative stick / limited roll response\nShading: near-full negative roll; dotted: motors zero; model moments are NOT measured torque')
fig.savefig(OUT/'roll_tracking.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(12,8),layout='constrained');axes[0].semilogy(freq,praw,label='Primary gyro FIFO x PSD (2 kHz)');axes[0].set_xlim(0,600);axes[0].axvline(peak,color='r',ls='--',label=f'Main peak {peak:.1f} Hz');axes[0].set_xlabel('Hz');axes[0].legend();axes[0].set_title('Raw mechanical vibration; 19-32 s')
axes[1].semilogy(fctrl,pc/pc.max(),label='Controlled roll-rate PSD (normalized)');axes[1].semilogy(fctrl,ps/ps.max(),label='Servo 1 PSD (normalized)');axes[1].semilogy(fg,pg/pg.max(),label='Mocap approximate roll-rate PSD (normalized)');axes[1].set_xlim(0,20);axes[1].set_xlabel('Hz');axes[1].legend();axes[1].set_title('Persistent attitude / servo oscillation is a separate low-frequency band')
fig.savefig(OUT/'vibration_spectrum.png',dpi=150);plt.close(fig)
