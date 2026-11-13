from pathlib import Path
import os,json,re
os.environ['MPLCONFIGDIR']='/tmp/mpl-gyro-ekf'
import numpy as np
from scipy.signal import welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
z=np.load(root/'analysis/roll_20261105_unknown1/topics.npz');meta=json.load(open(root/'analysis/roll_20261105_unknown1/metadata.json'))
result={'source':meta['source'],'windows_seconds':{'prearm_not_verified_stationary':[2,7],'flight':[19,32]},'fifo':{},'ekf':{}}
fig,axs=plt.subplots(3,2,figsize=(13,10))
for idx in [0,1]:
 a=z[f'sensor_gyro_fifo_{idx}'];mask=np.arange(32)[None,:]<a['samples'][:,None]
 t=((a['timestamp_sample'][:,None]+np.arange(32)[None,:]*a['dt'][:,None])*1e-6)[mask]
 v=np.stack([(a[f]*a['scale'][:,None])[mask] for f in 'xyz'],axis=1)
 entry={'device_id':int(a['device_id'][0]),'dt_us':float(a['dt'][0]),'windows':{}}
 for name,(lo,hi) in result['windows_seconds'].items():
  k=(t>=lo)&(t<hi);dt=np.diff(t[k]);x=v[k];w={'std_rad_s':x.std(axis=0).tolist(),'mean_rad_s':x.mean(axis=0).tolist(),'max_abs_rad_s':np.max(abs(x),axis=0).tolist(),'spacing_us_p50_p99_max':(np.quantile(dt,[.5,.99,1])*1e6).tolist(),'sample_count':int(k.sum())};entry['windows'][name]=w
  if idx==0:
   grid=np.arange(lo,hi,.0005);vv=np.stack([np.interp(grid,t,v[:,j]) for j in range(3)],axis=1);f,p=welch(vv,fs=2000,nperseg=min(8192,len(grid)),axis=0)
   band=(f>=250)&(f<=350);w['peak_250_350_hz']=f[band][np.argmax(p[band],axis=0)].tolist();w['rms_250_350_rad_s']=np.sqrt(np.trapezoid(p[band],f[band],axis=0)).tolist()
   for j in range(3):axs[j,0].semilogy(f,p[:,j],label=name)
 result['fifo'][str(idx)]=entry
b=z['estimator_sensor_bias_0'];t=b['timestamp']/1e6
for j in range(3):
 axs[j,0].set(xlim=(0,700),ylabel=f'{"XYZ"[j]} PSD (rad/s)^2/Hz',xlabel='Hz');axs[j,0].legend();axs[j,0].grid()
 axs[j,1].plot(t,b['gyro_bias'][:,j],label='gyro bias');axs[j,1].axvspan(19,32,alpha=.15);axs[j,1].set(xlabel='Boot time (s)',ylabel=f'{"XYZ"[j]} bias rad/s');axs[j,1].grid()
fig.tight_layout();fig.savefig(out/'gyro_spectrum_bias.png',dpi=160)
a=z['estimator_status_0'];k=(a['timestamp']>=19e6)&(a['timestamp']<32e6)
for bit in [0,1,2,3,4,5,7,9,10,11,12,13,14,24,26,29,34,44]:result['ekf'][f'control_bit_{bit}_flight_fraction']=float(np.mean((a['control_mode_flags'][k]>>bit)&1))
for name in ['gyro_bias','accel_bias']:
 result['ekf'][name+'_flight_min']=np.min(b[name][(t>=19)&(t<32)],axis=0).tolist();result['ekf'][name+'_flight_max']=np.max(b[name][(t>=19)&(t<32)],axis=0).tolist()
p={s.split()[2]:float(s.split()[3]) for s in open(root/'log/机架1.params') if not s.startswith('#') and len(s.split())==5}
result['parameter_differences']={k:{'file':v,'log_last_recorded':meta['params'].get(k)} for k,v in p.items() if k.startswith(('EKF','IMU','SENS_IMU','CAL_GYRO')) and (k not in meta['params'] or not np.isclose(v,meta['params'][k],rtol=1e-5,atol=1e-7))}
(out/'metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
