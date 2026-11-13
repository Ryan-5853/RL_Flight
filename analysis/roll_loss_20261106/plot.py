from pathlib import Path
import os,csv
os.environ['MPLCONFIGDIR']='/tmp/mocap_plot_cache'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
P=Path(__file__).resolve().parent;x=np.load(P/'aligned.npz');rows=list(csv.DictReader(open('/home/ryan/TeleaiTrans_dev/Dock_ground/logs/flights/flight_20261106T035442_144Z_aircraft-01_1.csv')));q=np.array([[float(r['pose_q_'+k]) for k in 'wxyz'] for r in rows]);q[:,2:]*=-1;me=R.from_quat(q[:,[1,2,3,0]]).as_euler('xyz',degrees=True);fig,ax=plt.subplots(6,1,figsize=(14,14),sharex=True,layout='constrained')
for j in range(2):
 ax[j].plot(x['t'],x['euler'][:,j],label='EKF');ax[j].plot(x['ground_t'],me[:,j],label='Mocap FLU->FRD');ax[j].plot(x['t'],x['rc'][:,j]*(10 if j==0 else -10),ls='--',label='RC target');ax[j].set_ylabel(['Roll deg','Pitch deg'][j]);ax[j].legend(ncol=3)
for j in range(3):ax[2].plot(x['t'],x['servos'][:,j],label=f'Servo {j+1}')
ax[2].set_ylabel('Servo command');ax[2].legend(ncol=3)
for j in range(2):ax[3].plot(x['t'],x['motors'][:,j],label=f'Motor {j+1}')
ax[3].set_ylabel('Motor command');ax[3].legend(ncol=2)
ax[4].plot(x['t'],x['height'],label='Mocap height');ax[4].set_ylabel('Height m')
z=np.load(P/'topics.npz');s=z['estimator_status_flags_0'];ax[5].step(s['timestamp']/1e6,s['cs_ev_yaw'],where='post',label='External heading aiding');ax[5].set_ylim(-.1,1.1);ax[5].set_ylabel('EV active');ax[5].legend();ax[5].set_xlabel('FC boot time (s)')
for a in ax:a.grid(alpha=.25);a.axvline(23.189,color='red',ls='--');a.axvspan(14,16,color='orange',alpha=.15);a.set_xlim(4,23.4)
fig.suptitle('Roll range consumed by attitude offset: EKF tracks -10 deg while physical roll remains near zero')
fig.savefig(P/'roll_reference_offset.png',dpi=150)
