import json,csv
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
OUT=Path(__file__).resolve().parent
z=np.load(OUT/'topics.npz');meta=json.loads((OUT/'metadata.json').read_text())
def time(a):return a['timestamp']/1e6
def val(key,field,t):
 a=z[key];v=a[field];return np.interp(t,time(a),v) if v.ndim==1 else np.stack([np.interp(t,time(a),v[:,i]) for i in range(v.shape[1])],axis=-1)
a=z['vehicle_attitude_0'];at=time(a);angles=Rotation.from_quat(a['q'][:,[1,2,3,0]]).as_euler('xyz',degrees=True)
assert np.max(abs(np.linalg.norm(a['q'],axis=1)-1))<1e-4
for key in z.files:assert np.all(np.diff(z[key]['timestamp'].astype('i8'))>=0),key
t=np.arange(54,61.6,.005);roll=np.interp(t,at,angles[:,0]);pitch=np.interp(t,at,angles[:,1]);rate=val('vehicle_angular_velocity_0','xyz',t);accel=val('vehicle_acceleration_0','xyz',t);servos=val('actuator_servos_0','control',t)[:,:3];motors=val('actuator_motors_0','control',t)[:,:2];rc=np.stack([val('manual_control_setpoint_0',f,t) for f in ['roll','pitch','yaw','throttle']],axis=1)
metrics={'boot_time_s':True,'windows':{},'events':{}}
for lo,hi in [(52,55),(56.5,58),(58,59.8),(59.8,60.3),(60.3,60.8)]:
 w={}
 for k in ['sensor_accel_0','sensor_accel_1']:
  d=z[k];mask=(time(d)>=lo)&(time(d)<hi);v=np.array([d[f][mask] for f in 'xyz']).T
  w[k]={'mean':v.mean(0).tolist(),'std_sampled_m_s2':v.std(0).tolist(),'clip_sum':d['clip_counter'][mask].sum(0).tolist()}
 mask=(t>=lo)&(t<hi)
 if mask.any():w.update({'rate_mean_rad_s':rate[mask].mean(0).tolist(),'roll_start_end_deg':roll[mask][[0,-1]].tolist(),'servo1_negative_saturation_fraction':float(np.mean(servos[mask,0]<=-.99))})
 metrics['windows'][f'{lo}-{hi}']=w
for label,mask in [('first_motor_positive',(motors[:,0]>.001)),('first_servo1_negative_saturation',(servos[:,0]<=-.99)),('roll_over_30',(abs(roll)>30)),('roll_over_60',(abs(roll)>60)),('body_rate_over_1rad_s',(np.linalg.norm(rate,axis=1)>1))]:
 metrics['events'][label]=float(t[np.flatnonzero(mask)[0]])
with open(OUT/'timeline.csv','w') as f:
 w=csv.writer(f);w.writerow(['boot_s','roll_deg','pitch_deg','p_rad_s','q_rad_s','r_rad_s','ax','ay','az','servo1','servo2','servo3','motor1','motor2','rc_roll','rc_pitch','rc_yaw','rc_throttle'])
 for j in range(len(t)):w.writerow([t[j],roll[j],pitch[j],*rate[j],*accel[j],*servos[j],*motors[j],*rc[j]])
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2))
# Self-contained SVG; separate the pre-flip drift from subsequent violent motion.
panels=[('Onboard attitude (deg)',[(t,roll,'#e45756','roll'),(t,pitch,'#1976d2','pitch')],-10,90),('RC and servo outputs',[(t,rc[:,0],'#111','RC roll'),(t,servos[:,0],'#e45756','servo 1'),(t,servos[:,1],'#1976d2','servo 2'),(t,servos[:,2],'#46a34b','servo 3')],-1.1,1.1),('Final motor outputs and RC throttle',[(t,motors[:,0],'#e45756','motor 1'),(t,motors[:,1],'#1976d2','motor 2'),(t,(rc[:,3]+1)/2,'#111','pilot collective')],0,.65),('Raw sensor_accel Z (m/s2), decimated log samples',[(time(z['sensor_accel_0']),z['sensor_accel_0']['z'],'#1976d2','accel 0'),(time(z['sensor_accel_1']),z['sensor_accel_1']['z'],'#e45756','accel 1')],-45,25)]
s=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="1030" viewBox="0 0 1100 1030"><rect width="1100" height="1030" fill="white"/>']
for n,(title,series,lo,hi) in enumerate(panels):
 top=35+250*n;s.append(f'<text x="70" y="{top}" font-family="sans-serif" font-size="17">{title}</text><defs><clipPath id="p{n}"><rect x="70" y="{top+25}" width="985" height="175"/></clipPath></defs>')
 for v in np.linspace(lo,hi,5):
  yy=top+25+175*(hi-v)/(hi-lo);s.append(f'<path d="M70 {yy}H1055" stroke="#ddd"/><text x="10" y="{yy+4}" font-size="13">{v:.1f}</text>')
 for v in np.arange(55,61.1,.5):
  xx=70+(v-55)/5.8*985;s.append(f'<path d="M{xx} {top+25}v175" stroke="#eee"/><text x="{xx-12}" y="{top+220}" font-size="12">{v:.1f}</text>')
 for j,(ts,ys,col,label) in enumerate(series):
  pts=' '.join(f'{70+(tt-55)/5.8*985:.2f},{top+25+175*(hi-y)/(hi-lo):.2f}' for tt,y in zip(ts,ys) if 55<=tt<=60.8 and np.isfinite(y))
  s.append(f'<polyline clip-path="url(#p{n})" points="{pts}" fill="none" stroke="{col}" stroke-width="1.4"/><text x="{570+j*123}" y="{top}" fill="{col}" font-size="12">{label}</text>')
s.append('</svg>');(OUT/'crash_timeline.svg').write_text('\n'.join(s))
