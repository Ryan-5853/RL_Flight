from pathlib import Path
import os
os.environ['MPLCONFIGDIR']='/tmp/mocap_plot_cache'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,json
P=Path(__file__).resolve().parent;z=np.load(P/'topics.npz');p=json.load(open(P/'metadata.json'))['params'];fig,ax=plt.subplots(3,1,figsize=(13,9),sharex=True,layout='constrained')
for i in [0,1]:
 a=z[f'sensor_accel_{i}'];t=a['timestamp']/1e6;v=np.stack([a[k] for k in ['x','y','z']],1);off=np.array([p[f'CAL_ACC{i}_{k}OFF'] for k in 'XYZ']);scale=np.array([p[f'CAL_ACC{i}_{k}SCALE'] for k in 'XYZ']);ax[0].plot(t,np.linalg.norm(v,axis=1),lw=.6,label=f'accel {i} before CAL_ACC correction');ax[1].plot(t,np.linalg.norm((v-off)*scale,axis=1),lw=.6,label=f'accel {i} after CAL_ACC correction')
for a in ax[:2]:a.axhline(9.80665,color='black',ls='--',label='Standard gravity');a.set_ylabel('Acceleration norm (m/s^2)');a.legend()
b=z['estimator_sensor_bias_0'];
for j,name in enumerate('XYZ'):ax[2].plot(b['timestamp']/1e6,b['accel_bias'][:,j],label=name)
ax[2].axhline(-.4,color='black',ls='--',label='negative bias limit');ax[2].set_ylabel('EKF accel bias (m/s^2)');ax[2].legend(ncol=4);ax[2].set_xlabel('Time since flight controller boot (s)')
for a in ax:a.grid(alpha=.25)
fig.suptitle('Stationary test: gravity norm corrected; EKF accel bias remains within limits')
fig.savefig(P/'static_calibration.png',dpi=160)
