from pathlib import Path
import os,csv,json
os.environ['MPLCONFIGDIR']='/tmp/mocap_plot_cache'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
P=Path(__file__).resolve().parent;z=np.load(P/'topics.npz');x=np.load(P/'aligned.npz');b=np.load(P/'actuation.npz');q=z['vehicle_visual_odometry_0'];qe=R.from_quat(q['q'][:,[1,2,3,0]]).as_euler('xyz',degrees=True);qy=np.unwrap(np.deg2rad(qe[:,2]))*180/np.pi;rows=list(csv.DictReader(open('/home/ryan/TeleaiTrans_dev/Dock_ground/logs/flights/flight_20261106T033559_414Z_aircraft-02_1.csv')));off=json.load(open(P/'ground_alignment.json'))['flight_20261106T033559_414Z_aircraft-02_1']['offset_s'];gt=np.array([float(r['mono_ns'])*1e-9-off for r in rows]);height=np.array([float(r['position_z_m']) for r in rows]);esc=z['esc_status_0'];rpm=esc['esc']['esc_rpm'][:,:2].astype(float);ratio=rpm[:,0]/np.maximum(rpm[:,1],1);fig,ax=plt.subplots(7,1,figsize=(14,17),sharex=True,layout='constrained')
ax[0].plot(x['t'],x['yaw'],label='EKF unwrapped yaw');ax[0].plot(q['timestamp_sample']/1e6,qy,label='Mocap FRD yaw',alpha=.6);ax[0].set_ylabel('Yaw deg')
ax[1].plot(b['t'],b['gyro'][:,2]*180/np.pi,label='Measured r');ax[1].plot(b['t'],b['rcyaw']*.3*180/np.pi,label='RC yaw target (before filter)');ax[1].set_ylabel('Yaw rate deg/s')
ax[2].plot(b['t'],b['ratio'],label='Model speed ratio (pre-map command = 2*actuator)');ax[2].plot(esc['timestamp']/1e6,ratio,label='ESC RPM ch1/ch2');ax[2].axhline(1.15,color='red',ls='--',label='Model limit');ax[2].set_ylim(.8,1.4);ax[2].set_ylabel('Speed ratio')
for j in range(2):ax[3].plot(b['t'],b['motors'][:,j],label=f'Motor ch{j+1}')
ax[3].set_ylabel('Motor command')
ax[4].plot(b['t'],b['common'],label='Mean of three servo commands');ax[4].set_ylabel('Servo common')
ax[5].plot(gt,height,label='Mocap Z');ax[5].set_ylabel('Height m')
for j in range(2):ax[6].plot(x['t'],x['euler'][:,j],label=['Roll','Pitch'][j])
ax[6].set_ylabel('Tilt deg');ax[6].set_xlabel('FC boot time (s)')
for a in ax:
 a.axvspan(32,34,color='orange',alpha=.15);a.axvline(46.054,color='red',ls='--');a.grid(alpha=.25);a.legend(fontsize=8,loc='upper left',ncol=2);a.set_xlim(28,46.2)
fig.suptitle('Takeoff yaw spin: coherent mocap/EKF; neutral yaw stick initially; actuator allocation reaches limits')
fig.savefig(P/'takeoff_spin.png',dpi=150)
