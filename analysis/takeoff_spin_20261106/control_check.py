exec(open('analysis/takeoff_spin_20261106/analyze.py').read().split("print('TIMELINE')")[0])
import re
from scipy.optimize import minimize_scalar
h=Path('px4_trans/px4/src/modules/nn_control/LqiCoaxialCollectivePayload.hpp').read_text()
def arr(name):return np.array([float(x) for x in re.findall(r'([-+\d.e]+)f',re.search(r'float '+name+r'\[\d+\] = \{(.*?)\};',h,re.S).group(1))])
u=arr('kUpperTable').reshape(-1,2);l=arr('kLowerTable').reshape(-1,2)
aa=z['actuator_motors_0'];tt=aa['timestamp_sample']/1e6;motor=aa['control'][:,:2];su=np.interp(motor[:,0]*2,u[:,0],u[:,1]);sl=np.interp(motor[:,1]*2,l[:,0],l[:,1]);ratio=su/np.maximum(sl,1e-8);servo=interp('actuator_servos_0','control',tt)[:,:3];common=servo.mean(1);gw=interp('vehicle_angular_velocity_0','xyz',tt);rcyaw=interp('manual_control_setpoint_0','yaw',tt);yrate=np.interp(tt,t,yaw)
checks=[]
for lo,hi in [(30,31),(31,32),(32,33),(33,34),(34,35),(35,39),(39,44),(44,45.8)]:
 k=(tt>=lo)&(tt<hi)&np.isfinite(motor).all(1);r=ratio[k];cr=rcyaw[k];ww=gw[k,2];print('CONTROL',lo,hi,'yawdelta',np.interp(hi,t,yaw)-np.interp(lo,t,yaw),'rc p5medp95',np.percentile(cr,[5,50,95]),'rate mean',ww.mean()*180/np.pi,'motor med',np.median(motor[k],axis=0),'pred ratio med,max',np.percentile(r,[50,100]),'ratio_cap %',100*np.mean(r>1.149),'common med,p5p95',np.percentile(common[k],[5,50,95]));checks.append(dict(window=[lo,hi],yaw_change_deg=float(np.interp(hi,t,yaw)-np.interp(lo,t,yaw)),rc_yaw_p5p50p95=np.percentile(cr,[5,50,95]).tolist(),mean_r_deg_s=float(ww.mean()*180/np.pi),predicted_speed_ratio_p50p100=np.percentile(r,[50,100]).tolist(),predicted_speed_ratio_cap_percent=float(100*np.mean(r>1.149)),servo_common_p5p50p95=np.percentile(common[k],[5,50,95]).tolist()))
# Rate sign consistency is independent of EKF being directly constrained by mocap orientation.
mr=R.from_quat(v['q'][:,[1,2,3,0]]);mdt=np.diff(vt);mid=(vt[:-1]+vt[1:])/2;mw=(mr[:-1].inv()*mr[1:]).as_rotvec()/mdt[:,None];k=(mid>31)&(mid<45)&(mdt>.003)&(mdt<.04)
def fit(lag):
 gg=interp('vehicle_angular_velocity_0','xyz',mid[k]+lag);return np.mean((gg-mw[k])**2)
f=minimize_scalar(fit,bounds=(-.04,.04),method='bounded');gg=interp('vehicle_angular_velocity_0','xyz',mid[k]+f.x);print('BODYRATE lagms',f.x*1000,'rmse',np.sqrt(f.fun),'axis correlations',[np.corrcoef(gg[:,j],mw[k,j])[0,1] for j in range(3)],'slopes',[np.dot(gg[:,j],mw[k,j])/np.dot(gg[:,j],gg[:,j]) for j in range(3)])
print('peak r',np.percentile(gw[(tt>31)&(tt<35),2]*180/np.pi,[0,50,100]));print('dropouts',meta['dropouts_ms']);print('esc rpm range',np.max(z['esc_status_0']['esc']['esc_rpm'],axis=0) if 'esc_rpm' in z['esc_status_0']['esc'].dtype.names else z['esc_status_0']['esc'].dtype)
(P/'control_metrics.json').write_text(json.dumps({'windows':checks,'mocap_rate_lag_ms':float(f.x*1000),'mocap_gyro_axis_correlations':[float(np.corrcoef(gg[:,j],mw[k,j])[0,1]) for j in range(3)]},indent=2));np.savez(P/'actuation.npz',t=tt,motors=motor,servos=servo,ratio=ratio,common=common,rcyaw=rcyaw,gyro=gw)
