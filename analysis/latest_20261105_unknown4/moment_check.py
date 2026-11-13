exec(open(__file__.replace('moment_check.py','analyze.py')).read().split('# Verify original samples')[0])
# Reconstruct moments only at native ground servo timestamps, align attitude/rate/RC.
k=(nt>=15)&(nt<110)&v;tt=nt[k];qq=fs(tt);rcn=interp(arr('rc'),arr('rc_timestamp_us')*1e-6,tt);rrn=interp(arr('angular_velocity'),rt,tt);ee=qq.as_euler('xyz');target=R.from_euler('xyz',np.c_[rcn[:,0]*np.pi/18,-rcn[:,1]*np.pi/18,ee[:,2]]);er=(target.inv()*qq).as_rotvec();pp=-.0814316645*er[:,0];dd=-.0168643519*rrn[:,0];ii=mx[k]-pp-dd
out=[]
for lo,hi in [(20,30),(40,50),(60,70),(80,90),(100,110)]:
 mm=(tt>=lo)&(tt<hi);un=mm&(abs(mx[k])<.0088);nk=(nt>=lo)&(nt<hi);out.append({'start':lo,'end':hi,'valid_samples':int(mm.sum()),'unclipped_samples':int(un.sum()),'inferred_I_median':float(np.median(ii[un])),'P_mean':float(pp[mm].mean()),'D_std':float(dd[mm].std()),'estimated_speed_ratio_near_boundary_pct':float(100*np.mean((ratio[nk]<1/1.148)|(ratio[nk]>1.148)))})
print(json.dumps(out,indent=2));(O/'moment_check.json').write_text(json.dumps(out,indent=2))
