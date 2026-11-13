from pathlib import Path
import os,json,csv,collections,re
os.environ['MPLCONFIGDIR']='/tmp/rl-flight-matplotlib'
import numpy as np
from scipy.spatial.transform import Rotation as R,Slerp
from scipy.signal import butter,sosfiltfilt,welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent;ROOT=O.parents[1];G=Path('/home/ryan/TeleaiTrans_dev/Dock_ground/logs/flights');z={k:v for k,v in np.load(O/'topics.npz').items()};meta=json.load(open(O/'metadata.json'))
def read(session):
 base=G/session;pair=collections.defaultdict(list)
 for l in open(str(base)+'.pair.jsonl'):
  x=json.loads(l)
  if 'message_id'in x:pair[x['message_id']].append(x)
 off=np.array([x['host_mono_ns']*1e-9-x['data']['fields']['timestamp_us']*1e-6 for x in pair[19]])
 rows=list(csv.DictReader(open(str(base)+'.csv')));gt=np.array([float(r['mono_ns'])*1e-9-np.median(off) for r in rows]);gq=R.from_quat([[float(r['pose_q_'+f]) for f in ['x','y','z','w']] for r in rows]);pos=np.array([[float(r['position_'+f+'_m']) for f in ['x','y','z']] for r in rows]);return pair,rows,gt,gq,pos,off
pair,rows,gt,gq,pos,off=read('flight_20261105T075340_388Z_aircraft-01_3');pr,rr,t0,q0,p0,of0=read('flight_20261105T075332_848Z_aircraft-01_2')
all19=sorted(pair[19]+pr[19],key=lambda x:x['data']['fields']['timestamp_us']);fields=[x['data']['fields'] for x in all19];valid=np.array([np.linalg.norm(f['q'])>.9 for f in fields]);fields=[f for f,k in zip(fields,valid) if k]
def arr(name):return np.array([f[name] for f in fields],float)
qt=arr('attitude_sample_us')*1e-6;keep=np.r_[True,np.diff(qt)>0];fc=R.from_quat(arr('q')[keep][:,[1,2,3,0]]);qt=qt[keep]
gt=np.r_[t0,gt];gq=R.concatenate([q0,gq]);pos=np.r_[p0,pos];ix=np.argsort(gt);gt=gt[ix];gq=gq[ix];pos=pos[ix];good=np.r_[True,np.diff(gt)>1e-6];gt=gt[good];gq=gq[good];pos=pos[good]
t=np.arange(11.2,34.7,.01);fs=Slerp(qt,fc);gs=Slerp(gt,gq);fr=fs(t)
def interp(values,times,target=t):
 if values.ndim==1:return np.interp(target,times,values)
 return np.stack([np.interp(target,times,values[:,j]) for j in range(values.shape[1])],1)
rt=arr('rates_timestamp_us')*1e-6;rate=interp(arr('angular_velocity'),rt);rc=interp(arr('rc'),arr('rc_timestamp_us')*1e-6);serv=interp(arr('servos'),arr('servos_timestamp_us')*1e-6);mot=interp(arr('motors'),arr('motors_timestamp_us')*1e-6)
def lp(x,fc=2):return sosfiltfilt(butter(3,fc,fs=100,output='sos'),x,axis=0)
def unit(v):return v/np.linalg.norm(v,axis=-1,keepdims=True)
def pct(x):return np.percentile(x,[5,50,95],axis=0).tolist()
rate_g=(gq[:-1].inv()*gq[1:]).as_rotvec()/np.diff(gt)[:,None];tm=(gt[:-1]+gt[1:])/2;fly=(t>=17)&(t<34);best=None
for lag in np.arange(-.06,.061,.002):
 aa=lp(interp(rate_g,tm,t+lag));bb=lp(rate);B,_=R.align_vectors(aa[fly]-aa[fly].mean(0),bb[fly]-bb[fly].mean(0));ee=aa-B.apply(bb);score=np.mean((ee[fly]-ee[fly].mean(0))**2)
 if best is None or score<best[0]:best=(score,lag,B)
score,lag,B=best;mr=gs(t+lag);pre=(t>=11.2)&(t<13);gf=fr.inv().apply(np.tile([0.,0.,1.],(len(t),1)));res={'source':meta['source'],'main_ground_session':'flight_20261105T075340_388Z_aircraft-01_3','ground_t_range': [float(gt[0]),float(gt[-1])],'dropout_total_ms':sum(meta['dropouts_ms']),'dropout_count':len(meta['dropouts_ms']),'alignment':{'best_lag_s':float(lag),'B_rotvec_deg':(B.as_rotvec()*180/np.pi).tolist(),'rate_rmse':float(np.sqrt(score))},'body_alignment_sensitivity':[],'windows':[]}
Bp=R.from_rotvec(np.deg2rad(json.load(open(O.parent/'attitude_mocap_20261105_unknown2/metrics.json'))['body_alignment_rotvec_deg']))
for name,bb in [('current_fit',B),('previous_flight_fit',Bp),('FRD_FLU_x180',R.from_euler('x',180,degrees=True))]:
 vv=(mr*bb).apply(gf);v0=unit(vv[pre].mean(0));ss=unit(lp(vv,.5));ang=np.rad2deg(np.arccos(np.clip(ss@v0,-1,1)));res['body_alignment_sensitivity'].append({'B':name,'windows':[[lo,hi,*pct(ang[(t>=lo)&(t<hi)])] for lo,hi in [(17,20),(20,25),(25,30),(30,34)]]})
 if name=='current_fit':vertical=vv;tilt=ang;W=(fr[pre]*(mr[pre]*bb).inv()).mean();aligned=W*mr*bb;err=(aligned.inv()*fr).as_rotvec()*180/np.pi;aline=aligned.as_euler('xyz',degrees=True)
e=fr.as_euler('xyz',degrees=True);height=interp(pos[:,2],gt)
# Restricted native ground output inference. These are telemetry sample counts, not wall-clock or true moments.
nt=arr('servos_timestamp_us')*1e-6;ns=arr('servos');nm=arr('motors');common=ns.mean(1);table=np.array([[0,0],[.08,0],[.1,446.991],[.2,578.304],[.3,717.411],[.4,831.996],[.5,936.197],[.6,988.958],[.8,1093.479],[1,1200]]);nw=np.stack([np.interp(2*nm[:,j],table[:,0],table[:,1]) for j in range(2)],1);ratio=nw[:,0]/np.maximum(nw[:,1],1);v=(common>-.0795)&(common<-.01)&(ratio>1/1.148)&(ratio<1.148)&(abs(ns).max(1)<.999);mx=(ns[:,0]-common)*(-common/.08)/72.0721436
for lo,hi in [(11.2,13),(17,20),(20,25),(25,30),(30,34)]:
 k=(t>=lo)&(t<hi);nk=(nt>=lo)&(nt<hi);vk=nk&v
 res['windows'].append({'start':lo,'end':hi,'target_rp_mean_deg':np.c_[rc[k,0]*10,-rc[k,1]*10].mean(0).tolist(),'fc_rp_mean_deg':e[k,:2].mean(0).tolist(),'mocap_aligned_rp_mean_deg':aline[k,:2].mean(0).tolist(),'slow_tilt_change_p5_p50_p95_deg':pct(tilt[k]),'full_error_rotvec_mean_deg':err[k].mean(0).tolist(),'stick_p5_p50_p95':pct(rc[k]),'servo_p5_p50_p95':pct(serv[k]),'motor_p5_p50_p95':pct(mot[k]),'rate_std':rate[k].std(0).tolist(),'height_p5_p50_p95':pct(height[k]),'moment_valid_samples':int(vk.sum()),'roll_negative_cap_pct_valid':float(100*np.mean(mx[vk]<-.00895)) if vk.any() else None,'roll_moment_mean_valid':float(mx[vk].mean()) if vk.any() else None})
# Verify original samples, not interpolation through ULog holes.
checks={}
for key,tsfield,value in [('vehicle_attitude_0','attitude_sample_us','q'),('vehicle_angular_velocity_0','rates_timestamp_us','angular_velocity'),('actuator_servos_0','servos_timestamp_us','servos')]:
 a=z[key];at=a['timestamp_sample'] if key=='vehicle_attitude_0' else a['timestamp'];st=arr(tsfield);j=np.searchsorted(at,st);j=np.clip(j,1,len(at)-1);j=np.where(abs(at[j-1].astype(float)-st)<abs(at[j].astype(float)-st),j-1,j);ok=abs(at[j].astype(float)-st)<500
 if value=='q':dot=abs(np.sum(a['q'][j[ok]]*arr(value)[ok],1));checks[key]={'matched_within_0p5ms':int(ok.sum()),'q_dot_min':float(dot.min())}
 else:
  actual=a['xyz'][j[ok]] if value=='angular_velocity' else a['control'][j[ok],:3];checks[key]={'matched_within_0p5ms':int(ok.sum()),'rmse':float(np.sqrt(np.nanmean((actual-arr(value)[ok])**2)))}
res['ulog_ground_checks']=checks
res['sampling']={}
for key in ['vehicle_attitude_0','actuator_servos_0','vehicle_angular_velocity_0']:
 a=z[key];tt=a['timestamp']*1e-6;k=(tt>=17)&(tt<34);d=np.diff(tt[k]);res['sampling'][key]={'rows_17_34':int(k.sum()),'max_gap_s':float(d.max()),'gaps_above_50ms':int(sum(d>.05)),'covered_time_with_20ms_cap_fraction':float(np.minimum(d,.02).sum()/17)}
res['sampling']['ground_fc_attitude']={'max_gap_s_17_34':float(np.diff(qt[(qt>=17)&(qt<34)]).max()),'samples_17_34':int(sum((qt>=17)&(qt<34)))}
res['ground']={'control_state':dict(collections.Counter(r['control_state'] for r in rows)),'source':dict(collections.Counter(r['control_source'] for r in rows)),'pose_fresh_pct':100*np.mean([r['pose_fresh']=='1' for r in rows]),'max_pose_age_ms':max(float(r['pose_age_ms']) for r in rows),'invalid_internal_trace_all':all(x['data']['fields']['timestamp_us']==0 for x in pair[18])}
res['spectra']=[]
for name,pp in [('current',pair),('previous',read('flight_20261105T071243_578Z_aircraft-01_1')[0])]:
 ff=[x['data']['fields'] for x in pp[19]];tt=np.array([x['rates_timestamp_us']*1e-6 for x in ff]);vv=np.array([x['angular_velocity'] for x in ff]);st=np.array([x['servos_timestamp_us']*1e-6 for x in ff]);sv=np.array([x['servos'] for x in ff],float);ttt=np.arange(19,32,.02);vr=interp(vv,tt,ttt);vs=interp(sv,st,ttt);f,p=welch(vr,50,nperseg=256,axis=0);_,ps=welch(vs,50,nperseg=256,axis=0);b=(f>=4)&(f<=8);b2=(f>=1)&(f<=15);res['spectra'].append({'flight':name,'window':[19,32],'rate_4_8_rms':np.sqrt(np.trapezoid(p[b],f[b],axis=0)).tolist(),'servo_4_8_rms':np.sqrt(np.trapezoid(ps[b],f[b],axis=0)).tolist(),'rate_peak_1_15_hz':f[b2][np.argmax(p[b2],axis=0)].tolist(),'max_native_gap_19_32':float(np.diff(tt[(tt>=19)&(tt<32)]).max())})
a=z['estimator_status_0'];k=(a['timestamp']>=17e6)&(a['timestamp']<34e6);defs=(ROOT/'px4_trans/px4/msg/EstimatorStatus.msg').read_text();bits={n:int(b) for n,b in re.findall(r'uint8 (CS_\w+) = (\d+)',defs)};bits.update(gravity_vector=34,inertial_dead_reckoning=29);res['ekf']={'samples_17_34':int(k.sum()),'flag_fractions':{n:float(np.mean((a['control_mode_flags'][k]>>b)&1)) for n,b in bits.items()},'fault_values':np.unique(a['filter_fault_flags'][k]).tolist(),'reset_quat_values':np.unique(a['reset_count_quat'][k]).tolist()}
a=z['estimator_sensor_bias_0'];k=(a['timestamp']>=17e6)&(a['timestamp']<34e6)
for f in ['gyro_bias','accel_bias']:res['ekf'][f+'_min_median_max']=np.percentile(a[f][k],[0,50,100],axis=0).tolist()
res['clipping']={}
for i in [0,1]:
 a=z[f'vehicle_imu_status_{i}'];k=(a['timestamp']>=17e6)&(a['timestamp']<34e6);res['clipping'][str(i)]={'accel_first_last':a['accel_clipping'][k][[0,-1]].tolist(),'gyro_first_last':a['gyro_clipping'][k][[0,-1]].tolist()}
# Runtime diagnostics can distinguish telemetry/logger loss from module staleness, not prove actuator schedule.
diag=[]
for x in pair[16]:
 f=x['data']['fields']
 if 17e6<f['timestamp_us']<34e6:diag.append(f)
res['runtime_diag']={}
for name in ['logger_dropouts','logger_message_gaps','logger_rate_kb_s','run_interval_max_us','run_over_10ms','local_imu_age_us','local_imu_age_max_us','local_core_age_us','local_core_age_max_us','local_core_stale_events']:
 vals=[f[name] for f in diag if name in f];res['runtime_diag'][name]={'min':min(vals),'max':max(vals)} if vals else None
(O/'metrics.json').write_text(json.dumps(res,indent=2));np.savez_compressed(O/'aligned.npz',t=t,tilt=tilt,vertical=vertical,fr_q=fr.as_quat(),mocap_aligned_q=aligned.as_quat(),euler=e,mocap_aligned_euler=aline,rc=rc,servos=serv,motors=mot,rate=rate,height=height,err=err)
fig,ax=plt.subplots(6,1,figsize=(14,15),sharex=True,layout='constrained')
for j in range(2):ax[j].plot(t,e[:,j],label='Onboard');ax[j].plot(t,aline[:,j],label='Mocap aligned from preflight');ax[j].plot(t,rc[:,j]*(10 if j==0 else -10),'k--',label='Stick target');ax[j].set_ylabel(['Roll deg','Pitch deg'][j])
ax[2].plot(t,tilt,label='Slow tilt change vs preflight, independent of yaw drift');ax[2].plot(t,err[:,2],label='Relative rotation error z deg')
for j in range(3):ax[3].plot(t,serv[:,j],label=f'Servo {j+1}')
for j in range(2):ax[4].plot(t,mot[:,j],label=f'Motor {j+1}')
ax[5].plot(t,height,label='Mocap height m')
for aa in ax:aa.legend(fontsize=8,ncol=3);aa.grid(alpha=.25);aa.axvspan(11.2,13,color='green',alpha=.12)
ax[-1].set_xlabel('FC boot sample time s');fig.savefig(O/'comparison.png',dpi=150)
print(json.dumps(res,indent=2))
