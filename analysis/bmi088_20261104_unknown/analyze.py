from pathlib import Path
import csv,json
import numpy as np
from scipy.spatial.transform import Rotation
OUT=Path(__file__).resolve().parent
z=np.load(OUT/'topics.npz');meta=json.loads((OUT/'metadata.json').read_text())
def times(a):return a['timestamp']/1e6
def interp(k,f,t):
 a=z[k];v=a[f];return np.interp(t,times(a),v) if v.ndim==1 else np.stack([np.interp(t,times(a),v[:,i]) for i in range(v.shape[1])],axis=-1)
a=z['vehicle_attitude_0'];t=times(a);e=Rotation.from_quat(a['q'][:,[1,2,3,0]]).as_euler('xyz',degrees=True)
assert np.max(abs(np.linalg.norm(a['q'],axis=1)-1))<1e-4
for k in z.files:assert np.all(np.diff(z[k]['timestamp'].astype('i8'))>=0)
rate=interp('vehicle_angular_velocity_0','xyz',t);sv=interp('actuator_servos_0','control',t)[:,:3];mot=interp('actuator_motors_0','control',t)[:,:2];rc=np.stack([interp('manual_control_setpoint_0',f,t) for f in ['roll','pitch','yaw','throttle']],axis=-1)
# Model-domain allocation proxies, not actual measured actuator torque.
D=np.array([[72.0721436,0],[-36.0360718,-62.4163055],[-36.0360718,62.4163055]])
proxy=sv@np.linalg.pinv(D).T
native={};metrics={'dropouts_ms':meta['dropouts_ms'],'selection':{f:int(z['sensor_selection_0'][f][0]) for f in ['accel_device_id','gyro_device_id']},'fifo':{},'windows':{}}
for key in ['sensor_gyro_fifo_0','sensor_gyro_fifo_1']:
 a=z[key];valid=np.arange(32)[None,:]<a['samples'][:,None];tt=(a['timestamp_sample'][:,None]+np.arange(32)[None,:]*a['dt'][:,None])/1e6;v=np.stack([a[f]*a['scale'][:,None] for f in 'xyz'],axis=-1);native[key]=(tt[valid],v[valid]);metrics['fifo'][key]={'device_ids':np.unique(a['device_id']).tolist(),'batches':len(a),'samples_in_recorded_batches':int(a['samples'].sum()),'dt_us':np.unique(a['dt']).tolist(),'batch_spacing_us_percentiles':np.percentile(np.diff(a['timestamp_sample'].astype('i8')),[0,50,95,100]).tolist()}
for lo,hi in [(7.5,11),(11,13.3),(13.6,13.9),(14,14.2)]:
 k=(t>=lo)&(t<hi);w={'roll_endpoints_deg':e[k,0][[0,-1]].tolist(),'pitch_endpoints_deg':e[k,1][[0,-1]].tolist(),'rate_mean_rad_s':rate[k].mean(0).tolist(),'allocation_proxy_mean':proxy[k].mean(0).tolist(),'gyro':{}}
 for key,(tt,v) in native.items():
  mask=(tt>=lo)&(tt<hi);w['gyro'][key]={'mean_xyz_rad_s':v[mask].mean(0).tolist(),'std_xyz_rad_s':v[mask].std(0).tolist()}
 metrics['windows'][f'{lo}-{hi}']=w
# Locate matching ground session via FC boot timestamps; motion capture is stale.
ground=Path('/home/ryan/TeleaiTrans_dev/Dock_ground/logs/flights/flight_20261104T082142_587Z_aircraft-01_7')
r=list(csv.DictReader(open(str(ground)+'.csv')));summary=json.load(open(str(ground)+'.summary.json'));metrics['ground']={'session':ground.name,'frames':len(r),'pose_sequences':len(set(x['pose_sequence'] for x in r)),'pose_stale_frames':summary['pose']['stale_frames'],'pose_callbacks':summary['vrpn']['callbacks'],'minimum_pose_age_ms':min(float(x['pose_age_ms']) for x in r)}
with open(OUT/'timeline.csv','w') as f:
 w=csv.writer(f);w.writerow(['boot_s','roll_deg','pitch_deg','yaw_deg','p_rad_s','q_rad_s','r_rad_s','servo1','servo2','servo3','motor1','motor2','rc_roll','rc_pitch','rc_yaw','rc_throttle','allocation_roll_proxy','allocation_pitch_proxy'])
 for i in range(len(t)):w.writerow([t[i],*e[i],*rate[i],*sv[i],*mot[i],*rc[i],*proxy[i]])
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2))
panels=[('Onboard roll/pitch: stable motor-on stage, then rapid rotation',[(t,e[:,0],'#e45756','roll'),(t,e[:,1],'#1976d2','pitch')],-5,70),('Body angular velocity (rad/s)',[(t,rate[:,0],'#e45756','p'),(t,rate[:,1],'#1976d2','q')],-1,8),('Servo commands (software outputs)',[(t,sv[:,0],'#e45756','servo 1'),(t,sv[:,1],'#1976d2','servo 2'),(t,sv[:,2],'#46a34b','servo 3')],-1.1,1.1),('Final motor commands',[(t,mot[:,0],'#e45756','upper'),(t,mot[:,1],'#1976d2','lower')],0,.55)]
s=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="1030"><rect width="1100" height="1030" fill="white"/>']
for n,(title,series,lo,hi) in enumerate(panels):
 top=35+250*n;s.append(f'<text x="70" y="{top}" font-family="sans-serif" font-size="16">{title}</text><defs><clipPath id="p{n}"><rect x="70" y="{top+25}" width="985" height="175"/></clipPath></defs>')
 for v in np.linspace(lo,hi,5):
  yy=top+25+175*(hi-v)/(hi-lo);s.append(f'<path d="M70 {yy}H1055" stroke="#ddd"/><text x="10" y="{yy+4}" font-size="13">{v:.1f}</text>')
 for v in np.arange(7,14.6,1):
  xx=70+(v-7)/7.5*985;s.append(f'<path d="M{xx} {top+25}v175" stroke="#eee"/><text x="{xx-12}" y="{top+220}" font-size="12">{v:.0f}s</text>')
 for j,(ts,ys,col,label) in enumerate(series):
  pts=' '.join(f'{70+(tt-7)/7.5*985:.2f},{top+25+175*(hi-y)/(hi-lo):.2f}' for tt,y in zip(ts,ys) if 7<=tt<=14.5 and np.isfinite(y))
  s.append(f'<polyline clip-path="url(#p{n})" points="{pts}" fill="none" stroke="{col}" stroke-width="1.4"/><text x="{780+j*95}" y="{top}" fill="{col}" font-size="12">{label}</text>')
s.append('</svg>');(OUT/'flight_timeline.svg').write_text('\n'.join(s))
