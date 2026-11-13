from pathlib import Path
import numpy as np,json,csv,collections
from scipy.spatial.transform import Rotation as R,Slerp
from scipy.spatial import cKDTree
P=Path(__file__).resolve().parent;G=Path('/home/ryan/TeleaiTrans_dev/Dock_ground/logs/flights');z=np.load(P/'topics.npz');meta=json.load(open(P/'metadata.json'));a=z['vehicle_attitude_0'];t=a['timestamp_sample']/1e6;ar=R.from_quat(a['q'][:,[1,2,3,0]]);e=ar.as_euler('xyz',degrees=True);yaw=np.unwrap(np.deg2rad(e[:,2]))*180/np.pi;v=z['vehicle_visual_odometry_0'];vt=v['timestamp_sample']/1e6;vq=v['q'].copy();vq*=np.where(vq[:,0:1]<0,-1,1);tree=cKDTree(vq);out={};allrows=[];ground=[]
for session in ['flight_20261106T033559_414Z_aircraft-02_1','flight_20261106T033622_944Z_aircraft-02_2']:
 rows=list(csv.DictReader(open(G/(session+'.csv'))));mono=np.array([float(r['mono_ns'])*1e-9 for r in rows]);q=np.array([[float(r['pose_q_'+j]) for j in 'wxyz'] for r in rows]);q[:,2:]*=-1;q*=np.where(q[:,0:1]<0,-1,1);d,idx=tree.query(q);k=(d<2e-6);offset=mono[k]-vt[idx[k]];med=np.median(offset);k&=abs(mono-vt[idx]-med)<.1;med=np.median(mono[k]-vt[idx[k]]);gt=mono-med
 out[session]={'offset_s':float(med),'matching_samples':int(sum(k)),'offset_spread_p5p95_ms':np.percentile((mono[k]-vt[idx[k]]-med)*1000,[5,95]).tolist(),'time_range':gt[[0,-1]].tolist(),'sources':dict(collections.Counter(r['control_source'] for r in rows)),'deadman':dict(collections.Counter(r['deadman'] for r in rows))};allrows.extend(rows);ground.append((rows,gt));print(session,out[session])
 # compare q against EKF output, no heading alignment fitted
 good=(gt>t[0])&(gt<t[-1]);err=(Slerp(t,ar)(gt[good]).inv()*R.from_quat(q[good][:,[1,2,3,0]])).magnitude()*180/np.pi;print('ground-EKF rotation err deg',np.percentile(err,[50,95,100]))
def interp(name,field,tt):
 aa=z[name];at=aa['timestamp_sample']/1e6 if 'timestamp_sample' in aa.dtype.names else aa['timestamp']/1e6;val=aa[field];return np.interp(tt,at,val) if val.ndim==1 else np.stack([np.interp(tt,at,val[:,j]) for j in range(val.shape[1])],1)
print('TIMELINE')
for st in [27,28.65,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46]:
 j=np.argmin(abs(t-st));rt=np.array([t[j]]);m=interp('manual_control_setpoint_0','yaw',rt);th=interp('manual_control_setpoint_0','throttle',rt);w=interp('vehicle_angular_velocity_0','xyz',rt)[0]*180/np.pi;mot=interp('actuator_motors_0','control',rt)[0,:2];sv=interp('actuator_servos_0','control',rt)[0,:3];rr,gt=ground[0];r=rr[np.argmin(abs(gt-st))];print(round(t[j],3),'e',np.round(e[j],1),'yaw_unwrap',round(yaw[j],1),'rate',np.round(w,1),'rcyaw/thr',np.round([m[0],th[0]],3),'mot',np.round(mot,3),'serv',np.round(sv,3),'height',r['position_z_m'],'gs',r['gs_mode_active'])
for lo,hi in [(24,28),(28.65,31),(31,34),(34,38),(38,42),(42,46)]:
 k=(t>=lo)&(t<hi);print('window',lo,hi,'yawchange',yaw[k][-1]-yaw[k][0]);aid=z['estimator_aid_src_ev_att_0'];kk=(aid['timestamp']/1e6>=lo)&(aid['timestamp']/1e6<hi);print('fusion',np.mean(aid['fused'][kk]),'maxinnovdeg',np.max(abs(aid['innovation'][kk]),axis=0)*180/np.pi)
(P/'ground_alignment.json').write_text(json.dumps(out,indent=2));np.savez(P/'aligned.npz',t=t,euler=e,yaw=yaw,ground_t=ground[0][1])
