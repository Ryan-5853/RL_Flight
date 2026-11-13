exec(open(__file__.replace('alignment_check.py','analyze.py')).read().split('# Independent body-frame alignment')[0])
from scipy.signal import sosfiltfilt,butter
mask=(t>=12)&(t<80);out=[]
for band in [(0,2),(.5,10),(2,12),(4,9)]:
 def filt(x):return sosfiltfilt(butter(3,band[1] if band[0]==0 else band,fs=100,btype='lowpass' if band[0]==0 else 'bandpass',output='sos'),x,axis=0)
 target=filt(gyro);best=None
 for lag in np.arange(-.05,.051,.001):
  bg=np.stack([np.interp(t+lag,gmid,grate[:,j]) for j in range(3)],1);aa=filt(bg);B,_=R.align_vectors(aa[mask]-aa[mask].mean(0),target[mask]-target[mask].mean(0));err=target-B.inv().apply(aa);score=np.mean((err[mask]-err[mask].mean(0))**2)
  if best is None or score<best[0]:best=(score,lag,B)
 score,lag,B=best;fc=fslerp(t);mg=gslerp(t+lag);gfc=fc.inv().apply(np.tile([0.,0.,1.],(len(t),1)));vertical=(mg*B).apply(gfc);pre=(t>=5)&(t<7.5);v0=unit(vertical[pre].mean(0));sl=unit(lp(vertical,.5));d=np.rad2deg(np.arccos(np.clip(sl@v0,-1,1)));entry={'band':band,'lag':lag,'B_rotvec_deg':(B.as_rotvec()*180/np.pi).tolist(),'rate_rmse':float(np.sqrt(score)),'tilt_windows':[]}
 for lo,hi in [(20,30),(40,50),(70,80)]:
  k=(t>=lo)&(t<hi);entry['tilt_windows'].append([lo,hi,*pct(d[k])])
 out.append(entry)
print(json.dumps(out,indent=2));(O/'alignment_sensitivity.json').write_text(json.dumps(out,indent=2))
