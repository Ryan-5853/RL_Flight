from pathlib import Path
import os,json
os.environ['MPLCONFIGDIR']='/tmp/rl-flight-matplotlib'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent;z=np.load(O/'topics.npz');b=z['estimator_sensor_bias_0'];a=z['vehicle_acceleration_0'];t=a['timestamp']*1e-6;bt=b['timestamp']*1e-6;bias=np.stack([np.interp(t,bt,b['accel_bias'][:,j]) for j in range(3)],1);k=(t>=4.5)&(t<6);raw=z['sensor_accel_0'];rk=(raw['timestamp']>=4.5e6)&(raw['timestamp']<6e6);rv=np.c_[raw['x'],raw['y'],raw['z']][rk]
r={'preflight_window_s':[4.5,6],'control_acceleration_samples':int(k.sum()),'control_acceleration_mean':a['xyz'][k].mean(0).tolist(),'norm_control_acceleration_mean':float(np.linalg.norm(a['xyz'][k].mean(0))),'approx_calibrated_prebias_mean':(a['xyz'][k]+bias[k]).mean(0).tolist(),'norm_approx_calibrated_prebias_mean':float(np.linalg.norm((a['xyz'][k]+bias[k]).mean(0))),'raw_sensor_acceleration_mean':rv.mean(0).tolist(),'norm_raw_sensor_mean':float(np.linalg.norm(rv.mean(0))),'limitation':'Adding interpolated online bias back to filtered control-layer acceleration is approximate; not native EKF IMU trace or exact calibration reconstruction.'}
(O/'sensor_check.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
fig,ax=plt.subplots(2,1,figsize=(13,7),sharex=True,layout='constrained')
for j in range(3):ax[0].plot(bt,b['accel_bias'][:,j],label='Accel '+ 'xyz'[j]);ax[1].plot(bt,b['gyro_bias'][:,j],label='Gyro '+ 'xyz'[j])
ax[0].axhline(-.4,color='r',ls='--');ax[0].axhline(.4,color='r',ls='--');ax[0].set_ylabel('Accel bias m/s2');ax[1].set_ylabel('Gyro bias rad/s');ax[1].set_xlabel('FC boot seconds')
for aa in ax:aa.set_xlim(4,35);aa.grid(alpha=.25);aa.legend(ncol=3)
fig.savefig(O/'bias.png',dpi=150)
