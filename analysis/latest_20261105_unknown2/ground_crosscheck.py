exec(open('analysis/latest_20261105_unknown2/analyze.py').read().split('off=[]')[0].replace('O=Path(__file__).resolve().parent',"O=Path('/home/ryan/RL_Flight/analysis/latest_20261105_unknown2')"))
from scipy.signal import coherence
rows=list(csv.DictReader(open(str(base)+'.csv')));off=[];qq=[];tt=[]
for l in open(str(base)+'.pair.jsonl'):
 x=json.loads(l)
 if x.get('message_id')==19:
  f=x['data']['fields'];off.append(x['host_mono_ns']/1e9-f['timestamp_us']/1e6)
gt=np.array([float(r['mono_ns'])/1e9-np.median(off) for r in rows]);gr=R.from_quat([[float(r['pose_q_'+f]) for f in ['x','y','z','w']] for r in rows]);gg=gr.as_euler('xyz',degrees=True)
q=ip(z,'vehicle_attitude_0','q',gt);q/=np.linalg.norm(q,axis=1)[:,None];fr=R.from_quat(q[:,[1,2,3,0]]);relative=gr.inv()*fr
print('orientation relative (mocap inverse onboard) rotvec mean/std deg')
for lo,hi in [(12,20),(20,30),(40,50),(60,70),(70,80)]:
 k=(gt>=lo)&(gt<hi);rv=relative[k].as_rotvec()*180/np.pi;print(lo,rv.mean(0),rv.std(0))
for lo,hi in [(19,32),(40,53),(67,80)]:
 t=np.arange(lo,hi,.01);dt=np.diff(gt);bg=(gr[:-1].inv()*gr[1:]).as_rotvec()/dt[:,None];br=np.stack([np.interp(t,(gt[:-1]+gt[1:])/2,bg[:,j]) for j in range(3)],1);r=ip(z,'vehicle_angular_velocity_0','xyz',t);f,p=welch(br,100,nperseg=512,axis=0);band=(f>=4)&(f<=8);fc,c=coherence(r[:,0],br[:,0],100,nperseg=256);print('mocap band',lo,np.sqrt(np.trapezoid(p[band],f[band],axis=0)),'coherence 6.6',c[np.argmin(abs(fc-6.6))])
a=z['manual_control_setpoint_0'];k=(a['timestamp']>12e6)&(a['timestamp']<80e6)
for f in ['roll','pitch','yaw','throttle']:print('stick',f,np.percentile(a[f][k],[0,5,50,95,100]))
a=z['actuator_servos_0'];k=(a['timestamp']>12e6)&(a['timestamp']<80e6);print('native servo percentiles',np.percentile(a['control'][k,:3],[0,5,50,95,100],axis=0))
a=z['esc_status_0'];print('esc dtype',a['esc'].dtype.names);print('esc example',a['esc'][1000,:2])
print('ground control state',collections.Counter(r['control_state'] for r in rows),'source',collections.Counter(r['control_source'] for r in rows),'deadman',collections.Counter(r['deadman'] for r in rows))
