"""Linear stability screen including servo lag, 50Hz ZOH and gyro LPF.
Gain multipliers are uncertainty assumptions, not identified from flight.
"""
import json,itertools
from pathlib import Path
import numpy as np
from scipy.linalg import expm
OUT=Path(__file__).resolve().parent
Ts=.02;inertia=.0005
plants=[]
for gain,tau,delay in itertools.product([.5,1.,1.5,2.,3.],[.04,.06,.10,.12],[0,1]):
 A=np.zeros((6,6));A[0,1]=1;A[1,2]=gain/inertia;A[2,2]=-1/tau;A[3,0]=1
 wn=2*np.pi*40;A[4,5]=1;A[5,4]=-wn**2;A[5,5]=-np.sqrt(2)*wn;A[5,1]=wn**2
 B=np.zeros((6,1));B[2]=1/tau
 aug=np.block([[A,B],[np.zeros((1,7))]]);disc=expm(aug*Ts);plants.append((gain,tau,delay,disc[:6,:6],disc[:6,6:]))
def evaluate(gains):
 kp,kd,ki=gains;K=np.array([[kp,0,0,ki,kd,0]])
 worst=0;cases=[]
 for gain,tau,delay,ad,bd in plants:
  closed=ad-bd@K if not delay else np.block([[ad,bd],[-K,np.zeros((1,1))]])
  poles=np.linalg.eigvals(closed);radius=float(max(abs(poles)));worst=max(worst,radius)
  cases.append({'gain_multiplier':gain,'servo_tau':tau,'extra_pwm_period_delay':delay,'pole_radius':radius})
 return worst,cases
candidates={'v2':(.0814316645,.0168643519,.0997526571),'a':(.04,.008,.025),'b':(.03,.006,.02),'c':(.025,.005,.0125),'d':(.02,.005,.01),'e':(.025,.006,.0125)}
results={}
for name,gains in candidates.items():
 worst,cases=evaluate(gains);results[name]={'kp_kd_ki':gains,'worst_pole_radius':worst,'unstable_assumed_cases':sum(c['pole_radius']>=1 for c in cases),'cases':cases};print(name,gains,'worst',round(worst,6),'unstable',results[name]['unstable_assumed_cases'])
(OUT/'linear_screen.json').write_text(json.dumps({'model':__doc__,'sample_period_s':Ts,'inertia':inertia,'results':results},indent=2))
