exec(open(__file__.replace('axis_check.py','analyze.py')).read().split('# Verify original samples')[0])
k=(nt>=10)&(nt<35)&v;tt=nt[k];qq=fs(tt);rcn=interp(arr('rc'),arr('rc_timestamp_us')*1e-6,tt);rrn=interp(arr('angular_velocity'),rt,tt);ee=qq.as_euler('xyz');target=R.from_euler('xyz',np.c_[rcn[:,0]*np.pi/18,-rcn[:,1]*np.pi/18,ee[:,2]]);er=(target.inv()*qq).as_rotvec();mom=np.c_[mx[k],(ns[k,2]-ns[k,1])*(-common[k]/.08)/124.832611];P=-.0814316645*er[:,:2];D=-.0168643519*rrn[:,:2];I=mom-P-D;out=[]
for lo,hi in [(10,15),(15,20),(20,25),(25,30),(30,33)]:
 kk=(tt>=lo)&(tt<hi);u=kk&(abs(mom).max(1)<.0088);out.append({'start':lo,'end':hi,'valid':int(kk.sum()),'Mx_My_mean':mom[kk].mean(0).tolist(),'negative_cap_pct':(100*np.mean(mom[kk]<-.00895,axis=0)).tolist(),'positive_cap_pct':(100*np.mean(mom[kk]>.00895,axis=0)).tolist(),'inferred_integral_median':np.median(I[u],axis=0).tolist(),'P_mean':P[kk].mean(0).tolist(),'D_std':D[kk].std(0).tolist()})
result={'moment_windows':out,'bias_windows':[],'pwm':[]}
a=z['estimator_sensor_bias_0'];at=a['timestamp']*1e-6
for lo,hi in [(4.5,6),(10,15),(15,20),(20,25),(25,30),(30,33)]:
 kk=(at>=lo)&(at<hi);result['bias_windows'].append({'start':lo,'end':hi,'count':int(kk.sum()),'accel_bias_mean':a['accel_bias'][kk].mean(0).tolist(),'gyro_bias_mean':a['gyro_bias'][kk].mean(0).tolist()})
result['bias_cap_first_s']={str(j):float(at[abs(a['accel_bias'][:,j])>.3999][0]) if np.any(abs(a['accel_bias'][:,j])>.3999) else None for j in range(3)}
old=json.load(open(O.parent/'latest_20261105_unknown4/metadata.json'))
for ch in [6,7,8]:
 pp=meta['params'];po=old['params'];result['pwm'].append({'channel':ch,'function':pp[f'PWM_MAIN_FUNC{ch}'],'new_minmax':[pp[f'PWM_MAIN_MIN{ch}'],pp[f'PWM_MAIN_MAX{ch}']],'new_armed_zero':(pp[f'PWM_MAIN_MIN{ch}']+pp[f'PWM_MAIN_MAX{ch}'])/2,'old_armed_zero':(po[f'PWM_MAIN_MIN{ch}']+po[f'PWM_MAIN_MAX{ch}'])/2})
# Independent mocap motion in the final full-stick section, no FC attitude used.
bg=interp(rate_g,tm,t+lag);result['full_stick_30_33']={'fc_rate_mean':rate[(t>=30)&(t<33)].mean(0).tolist(),'mocap_rate_fc_axes_mean':B.inv().apply(bg[(t>=30)&(t<33)]).mean(0).tolist(),'position_delta':(interp(pos,gt,np.array([33.]))-interp(pos,gt,np.array([30.])))[0].tolist()}
(O/'axis_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
