from pathlib import Path
import os
os.environ['MPLCONFIGDIR']='/tmp/mocap_plot_cache'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path(__file__).resolve().parent;z=np.load(P/'topics.npz');x=np.load(P/'aligned.npz');t=x['t'];dt=x['dt'];e=x['euler'];gap=dt>.03
fig,ax=plt.subplots(4,1,figsize=(14,10),sharex=True,layout='constrained')
for j,name in enumerate(['roll','pitch','yaw']):
 y=e[:,j].copy();y[np.flatnonzero(gap)+1]=np.nan;ax[0].plot(t,y,label=name,lw=.8)
ax[0].legend(ncol=3);ax[0].set_ylabel('Attitude (deg)')
ax[1].plot(t[1:][dt<.01],x['residual_deg'][dt<.01],'.',ms=1);ax[1].set_ylim(0,.3);ax[1].set_ylabel('Gyro mismatch (deg)\nonly intervals <10 ms')
ax[2].plot(t[1:],dt*1000,lw=.7);ax[2].set_ylabel('Logged attitude\ninterval (ms)')
s=z['estimator_status_0'];ax[3].step(s['timestamp']/1e6,((s['control_mode_flags']>>13)&1),where='post',label='External heading active');ax[3].set_ylim(-.1,1.2);ax[3].legend();ax[3].set_xlabel('Time since flight controller boot (s)')
for a in ax:
 for j in np.flatnonzero(gap):a.axvspan(t[j],t[j+1],color='grey',alpha=.15,lw=0)
 a.axvline(1.954496,color='red',ls='--',lw=.8);a.grid(alpha=.2)
fig.suptitle('Mocap attitude diagnostic: grey = missing log data; red = single startup yaw reset (16.72 deg)')
fig.savefig(P/'attitude_diagnostics.png',dpi=160)
