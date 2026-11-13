from pathlib import Path
import os,json
os.environ['MPLCONFIGDIR']='/tmp/mpl-compare'
import numpy as np
from scipy.signal import welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parent;root=out.parents[1]
result={};fig,ax=plt.subplots(3,1,figsize=(11,10))
for label,folder in [('before','roll_20261105_unknown1'),('after','gyro_compare_20261105_unknown3')]:
 z=np.load(root/'analysis'/folder/'topics.npz');m=json.load(open(root/'analysis'/folder/'metadata.json'));a=z['sensor_gyro_fifo_0'];valid=np.arange(32)[None,:]<a['samples'][:,None];t=((a['timestamp_sample'][:,None]+np.arange(32)*a['dt'][:,None])*1e-6)[valid];v=np.stack([(a[f]*a['scale'][:,None])[valid] for f in 'xyz'],axis=1)
 r={'dropout_count':len(m['dropouts_ms']),'dropout_total_s':sum(m['dropouts_ms'])/1000,'windows':{}}
 for lo,hi in [(19,32),(19,25),(25,32),(16,34)]:
  k=(t>=lo)&(t<hi);tt=t[k];vv=v[k];cuts=np.r_[0,np.where(np.diff(tt)>.0008)[0]+1,len(tt)];ps=[];weights=[]
  for start,end in zip(cuts[:-1],cuts[1:]):
   if end-start<512:continue
   grid=np.arange(tt[start],tt[end-1],.0005);xx=np.stack([np.interp(grid,tt[start:end],vv[start:end,j]) for j in range(3)],axis=1)
   if len(xx)<512:continue
   f,p=welch(xx,fs=2000,nperseg=512,noverlap=256,axis=0);ps.append(p);weights.append(1+(len(xx)-512)//256)
  p=np.average(ps,axis=0,weights=weights) if ps else None
  w={'raw_std_rad_s':vv.std(axis=0).tolist(),'raw_max_abs_rad_s':abs(vv).max(axis=0).tolist(),'sample_coverage':len(tt)/((hi-lo)*2000),'largest_gap_ms':float(np.diff(tt).max()*1000),'spectral_segment_count':len(ps),'spectral_windows':sum(weights)}
  if p is not None:
   for low,high in [(1,20),(200,400),(250,350)]:
    b=(f>=low)&(f<=high);w[f'rms_{low}_{high}']=np.sqrt(np.trapezoid(p[b],f[b],axis=0)).tolist()
   b=(f>=200)&(f<=400);w['peak_200_400_hz']=f[b][np.argmax(p[b],axis=0)].tolist()
   if (lo,hi)==(19,32):
    for j in range(3):ax[j].semilogy(f,p[:,j],label=label);ax[j].set(xlim=(0,650),ylabel=f'{"XYZ"[j]} PSD (rad/s)^2/Hz',xlabel='Hz');ax[j].legend();ax[j].grid()
  for topic,field in [('vehicle_angular_velocity_0','xyz'),('esc_status_0','esc'),('vehicle_imu_status_0','gyro_clipping'),('vehicle_imu_status_1','accel_clipping')]:
   a=z[topic];b=a[(a['timestamp']>=lo*1e6)&(a['timestamp']<hi*1e6)]
   if topic=='esc_status_0':w['rpm_median']=np.median(b[field]['esc_rpm'][:,:2],axis=0).tolist()
   elif field=='xyz':w['filtered_rate_std']=b[field].std(axis=0).tolist()
   else:w[topic+'_'+field+'_first_last']=[b[field][0].tolist(),b[field][-1].tolist()]
  r['windows'][f'{lo}-{hi}']=w
 for topic in ['vehicle_imu_status_0','vehicle_imu_status_1']:
  a=z[topic];r[topic]={f:[a[f][0].tolist(),a[f][-1].tolist()] for f in ['accel_clipping','gyro_clipping','gyro_error_count','accel_error_count']}
 a=z['estimator_status_0'];b=a[(a['timestamp']>=19e6)&(a['timestamp']<32e6)];r['ekf']={'gyro_ids':np.unique(b['gyro_device_id']).tolist(),'fault_flags':np.unique(b['filter_fault_flags']).tolist(),'reset_count_quat':np.unique(b['reset_count_quat']).tolist(),'hgt_test_ratio_max':float(np.nanmax(b['hgt_test_ratio']))}
 result[label]=r
fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);(out/'comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
