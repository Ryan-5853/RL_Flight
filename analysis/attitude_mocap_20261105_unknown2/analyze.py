from pathlib import Path
import os,json,csv,re
os.environ['MPLCONFIGDIR']='/tmp/rl-flight-matplotlib'
import numpy as np
from scipy.spatial.transform import Rotation as R,Slerp
from scipy.signal import butter,sosfiltfilt,savgol_filter
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent;SRC=O.parent/'latest_20261105_unknown2';ROOT=O.parents[1]
z={k:v for k,v in np.load(SRC/'topics.npz').items()};meta=json.load(open(SRC/'metadata.json'));base=Path('/home/ryan/TeleaiTrans_dev/Dock_ground/logs/flights/flight_20261105T071243_578Z_aircraft-01_1')
rows=list(csv.DictReader(open(str(base)+'.csv')));offset=[]
for line in open(str(base)+'.pair.jsonl'):
 x=json.loads(line)
 if x.get('message_id')==19:offset.append(x['host_mono_ns']*1e-9-x['data']['fields']['timestamp_us']*1e-6)
gt=np.array([float(r['mono_ns'])*1e-9-np.median(offset) for r in rows]);gq=np.array([[float(r['pose_q_'+f]) for f in ['x','y','z','w']] for r in rows]);gm=R.from_quat(gq);pos=np.array([[float(r['position_'+f+'_m']) for f in ['x','y','z']] for r in rows])
a=z['vehicle_attitude_0'];ft=a['timestamp_sample']*1e-6;fq=a['q'][:,[1,2,3,0]];keep=np.r_[True,np.diff(ft)>0];fslerp=Slerp(ft[keep],R.from_quat(fq[keep]));gslerp=Slerp(gt,gm)
t=np.arange(5,87,.01)
def ip(key,field,t=t):
 a=z[key];tt=a['timestamp_sample']*1e-6 if 'timestamp_sample' in a.dtype.names else a['timestamp']*1e-6;v=a[field]
 if v.ndim==1:return np.interp(t,tt,v)
 return np.stack([np.interp(t,tt,v[:,j]) for j in range(v.shape[1])],1)
def lp(v,fc=2):return sosfiltfilt(butter(3,fc,fs=100,output='sos'),v,axis=0)
def unit(v):return v/np.linalg.norm(v,axis=-1,keepdims=True)
def pct(v):return np.percentile(v,[5,50,95]).tolist()
fc=fslerp(t);gyro=ip('vehicle_angular_velocity_0','xyz');acc=ip('vehicle_acceleration_0','xyz');grate=(gm[:-1].inv()*gm[1:]).as_rotvec()/np.diff(gt)[:,None];gmid=(gt[:-1]+gt[1:])/2
# Independent body-frame alignment from rates. lag: query mocap at FC time + lag.
rate_mask=(t>=12)&(t<80);ratefit=[]
for lag in np.arange(-.10,.101,.002):
 bg=np.stack([np.interp(t+lag,gmid,grate[:,j]) for j in range(3)],1);aa=lp(bg);bb=lp(gyro);k=rate_mask
 bfit,_=R.align_vectors(aa[k]-aa[k].mean(0),bb[k]-bb[k].mean(0));bias=np.mean(bb[k]-bfit.inv().apply(aa[k]),0);err=bb-bfit.inv().apply(aa)-bias;ratefit.append((float(np.mean(err[k]**2)),lag,bfit,bias))
score,lag,B,bias=min(ratefit,key=lambda x:x[0]);mg=gslerp(t+lag);mb=mg*B;bg=np.stack([np.interp(t+lag,gmid,grate[:,j]) for j in range(3)],1)
# vertical vector in mocap-world coordinates. Constant for a correct estimate,
# regardless of fixed world yaw or fixed world tilt. No Euler-yaw subtraction.
gfc=fc.inv().apply(np.tile([0.,0.,1.],(len(t),1)));vertical=mb.apply(gfc)
pre=(t>=5)&(t<7.5);flight=(t>=12)&(t<80);v0=unit(np.mean(vertical[pre],0));vall=unit(np.mean(vertical[flight],0));vslow=unit(lp(vertical,.5))
angle=np.rad2deg(np.arccos(np.clip(vertical@v0,-1,1)));slowangle=np.rad2deg(np.arccos(np.clip(vslow@v0,-1,1)))
# Rotate world vertical reference into FC body coordinates through mocap.
gref=mb.inv().apply(np.tile(v0,(len(t),1)));tiltrot=np.cross(gref,gfc);tiltrot=np.rad2deg(tiltrot)
# Full fixed frame fit using preflight only, then test all flight without refitting.
W=(fc[pre]*mb[pre].inv()).mean();pred=W*mb;errfull=(pred.inv()*fc).as_rotvec()*180/np.pi
# Body B sensitivity from independent time segments and identity, no flight-attitude fitting.
sensitivity=[]
for name,lo,hi in [('early',12,40),('late',50,80),('all',12,80)]:
 k=(t>=lo)&(t<hi);aa=lp(bg);bb=lp(gyro);bf,_=R.align_vectors(aa[k]-aa[k].mean(0),bb[k]-bb[k].mean(0));vv=(mg*bf).apply(gfc);vpre=unit(vv[pre].mean(0));delta=np.rad2deg(np.arccos(np.clip(unit(lp(vv,.5))@vpre,-1,1)));sensitivity.append({'fit':name,'B_rotvec_deg':(bf.as_rotvec()*180/np.pi).tolist(),'flight_slow_tilt_p5_p50_p95':pct(delta[flight])})
# Apparent gravity from specific force. This is diagnostic, not EKF input trace.
accslow=lp(acc,.5);nacc=mb.apply(-unit(accslow));nacc=unit(nacc)
# Mocap position acceleration with 0.81s smoothing, scale/origin free; ignores marker lever arm.
pt=np.stack([np.interp(t+lag,gt,pos[:,j]) for j in range(3)],1);pa=savgol_filter(pt,81,3,deriv=2,delta=.01,axis=0)
# In FC body coordinates +z is down, in mocap world its direction is inferred -world_up.
# Local numeric quaternion frames have nearly matching axes; mocap height is +up,
# so map actual translation acceleration to numeric quaternion world with -identity.
# Do NOT assume this mapping for a physical prediction; only report magnitude for now.
result={'source':meta['source'],'method':'B maps FC body vectors into mocap rigid-body coordinates; vertical=(R_mocap B) R_fc^-1 ez; compare to fixed preflight vector. No fixed-yaw assumption.','lag_s':float(lag),'lag_definition':'mocap query time = FC time + lag; includes clock/receipt and filter delay','body_alignment_rotvec_deg':(B.as_rotvec()*180/np.pi).tolist(),'rate_fit_rmse_rad_s':float(np.sqrt(score)),'rate_fit_constant_bias_rad_s':bias.tolist(),'preflight_reference_mocap_world_vertical':v0.tolist(),'preflight_rate_rms':np.sqrt(np.mean(gyro[pre]**2,0)).tolist(),'preflight_mocap_rate_rms':np.sqrt(np.mean(bg[pre]**2,0)).tolist(),'preflight_position_range_m':np.ptp(pt[pre],axis=0).tolist(),'preflight_tilt_scatter_deg':pct(angle[pre]),'body_alignment_sensitivity':sensitivity,'windows':[]}
for lo,hi in [(5,7.5),(12,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(85,87)]:
 k=(t>=lo)&(t<hi);vm=unit(vertical[k].mean(0));ar=np.rad2deg(np.arccos(np.clip(vm@v0,-1,1)));nmean=unit(nacc[k].mean(0));app=np.rad2deg(np.arccos(np.clip(nmean@v0,-1,1)));res={'start':lo,'end':hi,'tilt_of_mean_vertical_deg':float(ar),'slow_tilt_p5_p50_p95_deg':pct(slowangle[k]),'vertical_mean':vm.tolist(),'body_tilt_error_xy_deg_mean':tiltrot[k,:2].mean(0).tolist(),'full_fixed_frame_error_mean_rotvec_deg':errfull[k].mean(0).tolist(),'apparent_gravity_tilt_deg':float(app),'accel_body_lowpass_mean':accslow[k].mean(0).tolist(),'accel_norm_lowpass_p5_p50_p95':pct(np.linalg.norm(accslow[k],axis=1)),'mocap_accel_world_mean':pa[k].mean(0).tolist()};result['windows'].append(res)
# Flexible constant frames fit is a lower-bound diagnostic: allowed to use all flight data.
# W R_m B = R_fc; if residual still varies, a fixed mounting/world transform is insufficient.
def residual(x,mask):
 pred=R.from_rotvec(x[:3])*mg[mask]*R.from_rotvec(x[3:]);return (pred.inv()*fc[mask]).as_rotvec().ravel()
fitmask=flight&(np.arange(len(t))%10==0);opt=least_squares(residual,np.r_[W.as_rotvec(),B.as_rotvec()],args=(fitmask,),loss='soft_l1',f_scale=.03,max_nfev=100);oo=residual(opt.x,flight).reshape(-1,3)*180/np.pi;result['all_flight_fixed_transform_fit']={'W_rotvec_deg':(opt.x[:3]*180/np.pi).tolist(),'B_rotvec_deg':(opt.x[3:]*180/np.pi).tolist(),'error_angle_p5_p50_p95_deg':pct(np.linalg.norm(oo,axis=1)),'caution':'fit uses all flight and includes yaw error; not an independent calibration'}
status=z['estimator_status_0'];sk=(status['timestamp']>=12e6)&(status['timestamp']<80e6);definitions=(ROOT/'px4_trans/px4/msg/EstimatorStatus.msg').read_text();bits={name:int(bit) for name,bit in re.findall(r'uint8 (CS_\w+) = (\d+)',definitions)}
result['ekf']={'control_flags_fraction':{name:float(np.mean((status['control_mode_flags'][sk]>>bit)&1)) for name,bit in bits.items()},'gravity_vector_fraction_bit34':float(np.mean((status['control_mode_flags'][sk]>>34)&1)),'inertial_dead_reckoning_fraction_bit29':float(np.mean((status['control_mode_flags'][sk]>>29)&1)),'filter_fault_values':np.unique(status['filter_fault_flags'][sk]).tolist(),'reset_quat_values':np.unique(status['reset_count_quat'][sk]).tolist(),'tilt_var_range':pct(ip('vehicle_local_position_0','tilt_var')[flight])}
for key in ['gyro_bias','accel_bias']:
 a=z['estimator_sensor_bias_0'];k=(a['timestamp']>=12e6)&(a['timestamp']<80e6);result['ekf'][key+'_min_max']=[a[key][k].min(0).tolist(),a[key][k].max(0).tolist()]
a=z['estimator_sensor_bias_0'];bt=a['timestamp']*1e-6;bk=(bt>=12)&(bt<80)
result['ekf']['accel_bias_limit']=float(a['accel_bias_limit'][bk][0])
result['ekf']['accel_bias_near_cap_first_s']={str(j):float(bt[bk&(abs(a['accel_bias'][:,j])>.3999)][0]) for j in [0,1]}
figb,ab=plt.subplots(2,1,figsize=(13,7),sharex=True,layout='constrained')
for j in range(3):ab[0].plot(bt,a['accel_bias'][:,j],label=f'{"xyz"[j]} accel bias');ab[1].plot(bt,a['gyro_bias'][:,j],label=f'{"xyz"[j]} gyro bias')
ab[0].axhline(.4,color='r',ls='--');ab[0].axhline(-.4,color='r',ls='--');ab[0].set_ylabel('m/s2');ab[1].set_ylabel('rad/s');ab[1].set_xlabel('FC boot seconds')
for axis in ab:axis.legend();axis.grid(alpha=.25);axis.set_xlim(5,87)
figb.savefig(O/'ekf_bias.png',dpi=150)
result['clipping']={} 
for i in [0,1]:
 a=z[f'vehicle_imu_status_{i}'];dt=np.diff(a['accel_clipping'].astype('i8'),axis=0);changed=np.any(dt>0,axis=1);result['clipping'][str(i)]={'device':int(a['accel_device_id'][0]),'accel_change_times_s':(a['timestamp'][1:][changed]*1e-6).tolist(),'final_accel_clipping':a['accel_clipping'][-1].tolist(),'final_gyro_clipping':a['gyro_clipping'][-1].tolist()}
(O/'metrics.json').write_text(json.dumps(result,indent=2))
np.savez_compressed(O/'aligned.npz',t=t,fc_q=fc.as_quat(),mocap_body_q=mb.as_quat(),vertical=vertical,preflight_vertical=v0,tilt_error_xy=tiltrot[:,:2],slow_tilt_deg=slowangle,apparent_vertical=nacc,accslow=accslow,mocap_acceleration=pa)
fig,ax=plt.subplots(5,1,figsize=(14,14),sharex=True,layout='constrained')
ax[0].plot(t,slowangle,label='Tilt change vs fixed preflight reference (0.5 Hz LP)');ax[0].plot(t,angle,alpha=.25,label='Instant tilt change');ax[0].set_ylabel('deg')
for j in range(2):ax[1].plot(t,tiltrot[:,j],label=f'Body tilt error {"xy"[j]} (small angle deg)')
for j in range(2):ax[2].plot(t,vslow[:,j],label=f'Estimated vertical in mocap world {"xy"[j]}');ax[2].plot(t,nacc[:,j],ls='--',alpha=.6,label=f'Accel apparent vertical {"xy"[j]}')
for j in range(3):ax[3].plot(t,errfull[:,j],label=f'Fixed frame error {"xyz"[j]} deg')
for j in range(3):ax[4].plot(t,accslow[:,j],label=f'Body specific force {"xyz"[j]} m/s2')
for a in ax:a.grid(alpha=.25);a.axvspan(5,7.5,color='green',alpha=.1);a.set_xlim(5,87);a.legend(fontsize=8,ncol=2)
ax[-1].set_xlabel('FC sample time seconds');fig.suptitle('Mocap vs onboard attitude: fixed body alignment from angular rates; preflight vertical reference');fig.savefig(O/'attitude_consistency.png',dpi=150)
print(json.dumps({k:v for k,v in result.items() if k!='clipping'},indent=2))
