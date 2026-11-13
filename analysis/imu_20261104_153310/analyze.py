from pathlib import Path
import csv,json
import numpy as np
from scipy.spatial.transform import Rotation
OUT=Path(__file__).resolve().parent
z=np.load(OUT/'topics.npz');meta=json.loads((OUT/'metadata.json').read_text());old=json.loads((OUT.parent/'crash_20261104_150954/metadata.json').read_text())
def time(a):return a['timestamp']/1e6
def interp(k,f,t):
 a=z[k];v=a[f]
 return np.interp(t,time(a),v) if v.ndim==1 else np.stack([np.interp(t,time(a),v[:,j]) for j in range(v.shape[1])],axis=-1)
a=z['vehicle_attitude_0'];t=time(a);e=Rotation.from_quat(a['q'][:,[1,2,3,0]]).as_euler('xyz',degrees=True)
assert np.max(abs(np.linalg.norm(a['q'],axis=1)-1))<1e-4
for k in z.files:assert np.all(np.diff(z[k]['timestamp'].astype('i8'))>=0)
rate=interp('vehicle_angular_velocity_0','xyz',t);acc=interp('vehicle_acceleration_0','xyz',t);mot=interp('actuator_motors_0','control',t)[:,:2];sv=interp('actuator_servos_0','control',t)[:,:3];rc=np.stack([interp('manual_control_setpoint_0',f,t) for f in ['roll','pitch','yaw','throttle']],axis=-1)
# Gravity-only proxy, valid as an angle reference only in static conditions.
acc_roll=np.degrees(np.arctan2(-acc[:,1],-acc[:,2]))
metrics={'windows':{},'changes_from_previous':{k:[old['params'].get(k),v] for k,v in meta['params'].items() if old['params'].get(k)!=v},'gyro_selection_changes':[]}
for lo,hi in [(10,35),(50.8,52.5),(53.8,55),(55,56.2),(56.5,57.5),(58.5,59)]:
 k=(t>=lo)&(t<hi);w={'roll_endpoints_deg':e[k,0][[0,-1]].tolist(),'roll_slope_deg_s':float(np.polyfit(t[k],e[k,0],1)[0]),'p_mean_rad_s':float(rate[k,0].mean()),'accel_roll_proxy_mean_deg':float(acc_roll[k].mean())}
 for key in ['sensor_accel_0','sensor_accel_1']:
  a=z[key];mask=(time(a)>=lo)&(time(a)<hi);v=np.array([a[f][mask] for f in 'xyz']).T;w[key]={'std_xyz_m_s2':v.std(0).tolist(),'clip_sum':a['clip_counter'][mask].sum(0).tolist()}
 metrics['windows'][f'{lo}-{hi}']=w
s=z['sensors_status_imu_0'];ids=s['gyro_device_id_primary'];idx=np.r_[0,np.flatnonzero(np.diff(ids.astype('i8'))!=0)+1]
metrics['gyro_selection_changes']=[{'boot_s':float(time(s)[i]),'device_id':int(ids[i])} for i in idx]
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2))
with open(OUT/'timeline.csv','w') as f:
 w=csv.writer(f);w.writerow(['boot_s','roll_deg','pitch_deg','yaw_deg','p_rad_s','q_rad_s','r_rad_s','accel_x','accel_y','accel_z','accel_gravity_only_roll_deg','motor1','motor2','servo1','servo2','servo3','rc_roll','rc_pitch','rc_yaw','rc_throttle'])
 for i in range(len(t)):w.writerow([t[i],*e[i],*rate[i],*acc[i],acc_roll[i],*mot[i],*sv[i],*rc[i]])
panels=[('Motor-off attitude: stable before handling',[(t,e[:,0],'#e45756','roll'),(t,e[:,1],'#1976d2','pitch')],10,35,-3,0),('Motor start: roll and gravity-only accel proxy',[(t,e[:,0],'#e45756','FC roll'),(t,acc_roll,'#1976d2','accel proxy')],52,57,-6,10),('Final motor commands',[(t,mot[:,0],'#e45756','motor 1'),(t,mot[:,1],'#1976d2','motor 2')],52,59,0,.55),('Raw accel Z: strong fluctuations after motor start',[(time(z['sensor_accel_0']),z['sensor_accel_0']['z'],'#1976d2','accel 0'),(time(z['sensor_accel_1']),z['sensor_accel_1']['z'],'#e45756','accel 1')],52,57,-45,25)]
s=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="1010"><rect width="1100" height="1010" fill="white"/>']
for n,(title,series,left,right,lo,hi) in enumerate(panels):
 top=30+250*n;s.append(f'<text x="70" y="{top}" font-family="sans-serif" font-size="17">{title}</text><defs><clipPath id="p{n}"><rect x="70" y="{top+25}" width="985" height="175"/></clipPath></defs>')
 for v in np.linspace(lo,hi,5):
  yy=top+25+175*(hi-v)/(hi-lo);s.append(f'<path d="M70 {yy}H1055" stroke="#ddd"/><text x="10" y="{yy+4}" font-size="13">{v:.1f}</text>')
 for v in np.linspace(left,right,6):
  xx=70+(v-left)/(right-left)*985;s.append(f'<path d="M{xx} {top+25}v175" stroke="#eee"/><text x="{xx-12}" y="{top+220}" font-size="12">{v:.1f}s</text>')
 for j,(ts,ys,col,label) in enumerate(series):
  pts=' '.join(f'{70+(tt-left)/(right-left)*985:.2f},{top+25+175*(hi-y)/(hi-lo):.2f}' for tt,y in zip(ts,ys) if left<=tt<=right and np.isfinite(y))
  s.append(f'<polyline clip-path="url(#p{n})" points="{pts}" fill="none" stroke="{col}" stroke-width="1.4"/><text x="{780+j*135}" y="{top}" fill="{col}" font-size="13">{label}</text>')
s.append('</svg>');(OUT/'imu_motor_comparison.svg').write_text('\n'.join(s));print(json.dumps(metrics,indent=2))
