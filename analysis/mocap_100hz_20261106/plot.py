from pathlib import Path
import os
os.environ['MPLCONFIGDIR']='/tmp/mocap_plot_cache'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
P=Path(__file__).resolve().parent;z=np.load(P/'topics.npz');x=np.load(P/'aligned.npz');v=z['vehicle_visual_odometry_0'];aid=z['estimator_aid_src_ev_att_0'];l=z['logger_status_0'];t=x['t'];mt=v['timestamp_sample']/1e6;me=R.from_quat(v['q'][:,[1,2,3,0]]).as_euler('xyz',degrees=True)
fig,ax=plt.subplots(6,1,figsize=(14,15),sharex=True,layout='constrained')
for j in range(3):
 ax[j].plot(t,x['euler'][:,j],label='EKF',lw=.9);ax[j].plot(mt,me[:,j],label='Mocap',lw=.8,alpha=.75);ax[j].set_ylabel(['Roll (deg)','Pitch (deg)','Yaw (deg)'][j]);ax[j].legend(loc='upper left',ncol=2)
for j in range(3):ax[3].plot(aid['timestamp_sample']/1e6,aid['innovation'][:,j]*180/np.pi,label=['x','y','z'][j],lw=.7)
ax[3].set_ylabel('Fusion innovation (deg)');ax[3].legend(ncol=3)
k=t[1:]>42;ax[4].plot(t[1:][k],x['residual_deg'][k],lw=.7);ax[4].set_ylabel('Gyro/attitude increment\nmismatch (deg)');ax[4].set_ylim(0,.25)
ax[5].plot(l['timestamp']/1e6,l['write_rate_kb_s'],label='Write rate (KiB/s)');ax[5].plot(l['timestamp']/1e6,l['buffer_used_bytes']/1024,label='Buffer used (KiB)');ax[5].legend();ax[5].set_xlabel('Time since flight controller boot (s)')
for a in ax:a.grid(alpha=.25);a.axvline(41.134211,color='red',ls='--',lw=.7)
fig.suptitle('100 Hz mocap: zero dropouts; yaw aligned once at mocap activation (41.134 s)')
fig.savefig(P/'attitude_mocap_diagnostics.png',dpi=150)
