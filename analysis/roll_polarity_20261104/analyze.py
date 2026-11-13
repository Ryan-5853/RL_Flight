import csv,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path('/home/ryan/TeleaiTrans_dev/Dock_ground/logs/flights')
OUT=Path(__file__).resolve().parent
results=[]
for stem in ['flight_20261104T064431_947Z_aircraft-01_2']:
 rows=list(csv.DictReader(open(ROOT/(stem+'.csv')))); t0=int(rows[0]['mono_ns'])
 t=np.array([(int(r['mono_ns'])-t0)/1e9 for r in rows]);q=np.array([[float(r['pose_q_'+k]) for k in 'xyzw'] for r in rows]); mr=Rotation.from_quat(q);me=mr.as_euler('xyz',degrees=True)
 diag=[]
 for l in open(ROOT/(stem+'.pair.jsonl')):
  x=json.loads(l)
  if x.get('message_id')==19 and x.get('aircraft')=='master':diag.append(x)
 dt=np.array([(x['host_mono_ns']-t0)/1e9 for x in diag]); f=[x['data']['fields'] for x in diag]
 fr=Rotation.from_quat(np.array([[z['q'][k] for k in [1,2,3,0]] for z in f]));fe=fr.as_euler('xyz',degrees=True)
 gyro=np.array([z['angular_velocity'] for z in f]);sf=np.array([z['acceleration'] for z in f]);rc=np.array([z['rc'] for z in f]);servo=np.array([z['servos'] for z in f]);motor=np.array([z['motors'] for z in f])
 # FLU body rates mapped to FRD: x unchanged, y/z reverse. Compare rates,
 # invariant to constant world yaw. Fixed marker-to-body mounting remains a caveat.
 unique=np.r_[True,np.diff([int(r['pose_sequence']) for r in rows])>0]; valid=unique & (np.array([float(r['pose_age_ms']) for r in rows])<30)
 tm=t[valid]; rm=mr[valid]; steps=np.diff(tm); pm=(rm[:-1].inv()*rm[1:]).as_rotvec()[:,0]/steps;tp=(tm[:-1]+tm[1:])/2
 metrics={}
 for a,b in [(5,8.5),(9,10),(10,11),(11,12),(12,13.3)]:
  k=(dt>=a)&(dt<b); km=(tp>=a)&(tp<b)
  if not k.any() or not km.any():continue
  moc=np.interp(dt[k],t,me[:,0]); metrics[f'{a}-{b}s']={'rc_roll_mean':float(rc[k,0].mean()),'fc_roll_mean_deg':float(fe[k,0].mean()),'mocap_roll_mean_deg':float(moc.mean()),'fc_minus_mocap_roll_deg':float((fe[k,0]-moc).mean()),'gyro_p_mean_deg_s':float(np.degrees(gyro[k,0].mean())),'mocap_p_mean_deg_s':float(np.degrees(pm[km].mean())),'specific_force_y_mean':float(sf[k,1].mean()),'servos_mean':servo[k].mean(0).tolist()}
 print(stem,json.dumps(metrics,indent=2))
 with open(OUT/(stem+'_aligned.csv'),'w') as fp:
  w=csv.writer(fp);w.writerow(['t_s','rc_roll','fc_roll_deg','mocap_roll_deg','fc_p_deg_s','mocap_p_deg_s','sf_y','servo1','servo2','servo3','motor1','motor2','pose_age_ms','x','y','z'])
  for j,tt in enumerate(dt):
   w.writerow([tt,rc[j,0],fe[j,0],np.interp(tt,t,me[:,0]),np.degrees(gyro[j,0]),np.degrees(np.interp(tt,tp,pm)),sf[j,1],*servo[j],*motor[j],*[np.interp(tt,t,[float(r[key]) for r in rows]) for key in ['pose_age_ms','position_x_m','position_y_m','position_z_m']]])
 results.append({'log':stem,'windows':metrics})
 if stem.endswith('_2'):
  # Simple standalone SVG plot, use only fresh motion capture samples.
  panels=[('Roll: FC vs motion capture (deg)',[(dt,fe[:,0],'#e45756','FC'),(t[valid],me[valid,0],'#1976d2','Mocap')],-30,35),('RC roll (normalized)',[(dt,rc[:,0],'#333','RC roll')],-1,1),('Motion capture position (m)',[(t[valid],np.array([float(r['position_y_m']) for r in rows])[valid],'#1976d2','world Y'),(t[valid],np.array([float(r['position_z_m']) for r in rows])[valid],'#46a34b','height')],-1.5,3)]
  svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="760" viewBox="0 0 1000 760"><rect width="1000" height="760" fill="white"/>']
  for n,(title,series,lo,hi) in enumerate(panels):
   top=35+n*240;svg.append(f'<text x="75" y="{top}" font-family="sans-serif" font-size="17">{title}</text>')
   for val in np.linspace(lo,hi,5):
    yy=top+25+170*(hi-val)/(hi-lo);svg.append(f'<path d="M75 {yy}H960" stroke="#ddd"/><text x="15" y="{yy+5}" font-size="13">{val:.1f}</text>')
   for xx in range(5,14):
    xp=75+(xx-5)/8.5*885;svg.append(f'<path d="M{xp} {top+25}v170" stroke="#eee"/><text x="{xp}" y="{top+215}" font-size="13">{xx}s</text>')
   for si,(ts,ys,color,label) in enumerate(series):
    pts=' '.join(f'{75+(tt-5)/8.5*885:.2f},{top+25+170*(hi-yy)/(hi-lo):.2f}' for tt,yy in zip(ts,ys) if 5<=tt<=13.5)
    svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.7"/><text x="{660+si*135}" y="{top}" fill="{color}" font-size="14">{label}</text>')
  svg.append('</svg>');(OUT/'latest_flight.svg').write_text('\n'.join(svg))
(OUT/'metrics.json').write_text(json.dumps(results,indent=2))
