"""Linear comparison: servo lag, 50Hz ZOH, 40Hz gyro LPF and explicit transport delay.
Uncertainty range is hypothetical, not identified. 20ms extra-delay stresses
are retained even though not all are stable; no flight-validation claim.
"""
from pathlib import Path
import json
import numpy as np,itertools
from scipy.linalg import expm
plants=[]
for gain,tau,delay in itertools.product([.5,1.,1.5,2.,3.],[.04,.06,.10,.12],[0,.002,.004,.010,.020]):
 A=np.zeros((6,6));A[0,1]=1;A[1,2]=gain/.0005;A[2,2]=-1/tau;A[3,0]=1;wn=2*np.pi*40;A[4,5]=1;A[5,4]=-wn**2;A[5,5]=-np.sqrt(2)*wn;A[5,1]=wn**2;B=np.zeros((6,1));B[2]=1/tau
 def disc(t):
  d=expm(np.block([[A,B],[np.zeros((1,7))]])*t);return d[:6,:6],d[:6,6:]
 ad,bd=disc(.02);ar,br=disc(.02-delay);_,bp=disc(delay);plants.append((gain,tau,delay,ad,br,ar@bp))
results={}
for name,gains in [('v2',(.0814316645,.0168643519,.0997526571)),('v3',(.012,.004,.020))]:
 K=np.array([[gains[0],0,0,gains[2],gains[1],0]]);out=[]
 for g,t,d,ad,br,bp in plants:
  c=np.block([[ad-br@K,bp],[-K,np.zeros((1,1))]]);r=max(abs(np.linalg.eigvals(c)));out.append((d,r,g,t))
 results[name]={'kp_kd_ki':gains,'delay_groups':[{'extra_delay_s':d,'worst_pole_radius':float(max(x[1] for x in out if x[0]==d)),'unstable_cases':int(sum(x[1]>=1 for x in out if x[0]==d))} for d in [0,.002,.004,.01,.02]],'cases':[{'delay_s':d,'radius':float(r),'gain_multiplier':g,'servo_tau_s':t} for d,r,g,t in out]}
 print(name,results[name]['delay_groups'])

assert all(g['unstable_cases']==0 for g in results['v3']['delay_groups'] if g['extra_delay_s']<=.004)
w=2*np.pi*5.6
amplitudes={name:float(abs(complex(kp,kd*w-ki/w))) for name,(kp,kd,ki) in [('v2',(.0814316645,.0168643519,.0997526571)),('v3',(.012,.004,.020))]}
(Path(__file__).resolve().parent/'pwm_delay_screen.json').write_text(json.dumps({'scope':__doc__,'results':results,'controller_magnitude_at_5p6hz':amplitudes,'v3_vs_v2_db':float(20*np.log10(amplitudes['v3']/amplitudes['v2']))},indent=2))
