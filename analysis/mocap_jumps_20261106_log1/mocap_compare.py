from pathlib import Path
import numpy as np,json
from scipy.spatial.transform import Rotation as R,Slerp
from scipy.optimize import minimize_scalar
P=Path(__file__).resolve().parent;z=np.load(P/'topics.npz');a=z['vehicle_attitude_0'];v=z['vehicle_visual_odometry_0'];g=z['vehicle_angular_velocity_0'];aid=z['estimator_aid_src_ev_att_0'];
t=a['timestamp_sample']/1e6;mt=v['timestamp_sample']/1e6;gt=g['timestamp_sample']/1e6;ar=R.from_quat(a['q'][:,[1,2,3,0]]);mr=R.from_quat(v['q'][:,[1,2,3,0]]);mid=(mt[1:]+mt[:-1])/2;mdt=np.diff(mt);mw=(mr[:-1].inv()*mr[1:]).as_rotvec()/mdt[:,None];k=(mid>5)&(np.linalg.norm(mw,axis=1)>.15)&(mdt>.005)&(mdt<.05)
def score(lag):
 gw=np.stack([np.interp(mid[k]+lag,gt,g['xyz'][:,j]) for j in range(3)],axis=1);return np.mean((gw-mw[k])**2)
res=minimize_scalar(score,bounds=(-.15,.15),method='bounded');lag=res.x
print('mocap timestamp to matching gyro lag s',lag,'RMSE',np.sqrt(res.fun),'zero RMSE',np.sqrt(score(0)))
print('odom interval ms',np.percentile(mdt*1000,[0,50,95,99,100]));print('odom age ms',np.percentile((v['timestamp'].astype(float)-v['timestamp_sample'])/1000,[0,50,95,99,100]));print('att interval ms',np.percentile(np.diff(t)*1000,[0,50,99,100]));print('aid age ms',np.percentile((aid['timestamp'].astype(float)-aid['timestamp_sample'])/1000,[0,50,95,100]));
for shift in [0,lag]:
 sel=(mt+shift>=t[0])&(mt+shift<=t[-1])&(mt>5);err=(Slerp(t,ar)(mt[sel]+shift).inv()*mr[sel]).as_rotvec()*180/np.pi;print('att minus mocap shift',shift,'err abs deg p50 p95 max',np.percentile(np.linalg.norm(err,axis=1),[50,95,100]));print('axis abs p95',np.percentile(np.abs(err),95,axis=0))
short=np.diff(t)<.01;drot=(ar[:-1].inv()*ar[1:]).magnitude()*180/np.pi
for j in np.argsort(drot)[-10:][::-1]:print('max step',t[j+1],np.diff(t)[j]*1000,drot[j])
print('rejects',[(float(r['timestamp']/1e6),r['innovation'].tolist(),r['test_ratio'].tolist()) for r in aid[aid['innovation_rejected']]])
l=z['logger_status_0'];print('logger rate median max',np.median(l['write_rate_kb_s']),max(l['write_rate_kb_s']),'dropouts',max(l['dropouts']),'buffer used max',max(l['buffer_used_bytes']),'buffer size',np.unique(l['buffer_size_bytes']))
(P/'comparison_metrics.json').write_text(json.dumps({'matching_gyro_lag_s':float(lag),'rate_rmse_rad_s':float(np.sqrt(res.fun)),'logger_write_rate_median_kib_s':float(np.median(l['write_rate_kb_s'])),'max_buffer_used_bytes':int(max(l['buffer_used_bytes']))},indent=2))
