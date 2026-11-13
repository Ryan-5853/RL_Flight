from pathlib import Path
import json,numpy as np
from scipy.spatial.transform import Rotation as R
P=Path(__file__).resolve().parent;m=json.load(open(P/'metadata.json'));p=m['params'];z=np.load(P/'topics.npz');rows=[]
for i in [0,1]:
 a=z[f'sensor_accel_{i}'];k=(a['timestamp']>=10e6)&(a['timestamp']<45e6);raw=np.stack([a[x][k] for x in ['x','y','z']],1);off=np.array([p[f'CAL_ACC{i}_{x}OFF'] for x in 'XYZ']);scale=np.array([p[f'CAL_ACC{i}_{x}SCALE'] for x in 'XYZ']);cal=(raw-off)*scale
 rows.append(dict(instance=i,device_id=int(a['device_id'][0]),raw_mean=raw.mean(0).tolist(),raw_std=raw.std(0).tolist(),raw_norm_mean=float(np.linalg.norm(raw,axis=1).mean()),offset=off.tolist(),scale=scale.tolist(),calibrated_mean_before_board_rotation=cal.mean(0).tolist(),calibrated_norm_mean=float(np.linalg.norm(cal,axis=1).mean())))
b=z['estimator_sensor_bias_0'];t=b['timestamp']/1e6;lim=np.flatnonzero(abs(b['accel_bias'][:,2])>=.399);v=z['vehicle_acceleration_0'];vt=v['timestamp']/1e6;bias=np.stack([np.interp(vt,t,b['accel_bias'][:,j]) for j in range(3)],1);k=(vt>=10)&(vt<45);before=v['xyz']+bias
r={'window_s':[10,45],'accels':rows,'selected_accel_id':int(z['sensor_selection_0']['accel_device_id'][0]),'bias_limit_first_s':float(t[lim[0]]),'end_bias':b['accel_bias'][-1].tolist(),'vehicle_acceleration_mean':v['xyz'][k].mean(0).tolist(),'vehicle_acceleration_norm':float(np.linalg.norm(v['xyz'][k],axis=1).mean()),'vehicle_accel_plus_bias_norm':float(np.linalg.norm(before[k],axis=1).mean())}
a=z['estimator_aid_src_ev_att_0'];r['fusion_success_fraction']=float(np.mean(a['fused']));r['max_innovation_deg']=(np.max(abs(a['innovation']),axis=0)*180/np.pi).tolist();s=z['estimator_status_flags_0'];r['rest_fraction']=float(np.mean(s['cs_vehicle_at_rest']));(P/'metrics.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
