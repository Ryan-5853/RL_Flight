"""Frozen-input controller replay; does NOT predict an alternate flight."""
from pathlib import Path
import re,ctypes,tempfile,subprocess,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
original=ROOT/'analysis/crash_20261105_unknown'
baseline=np.load(original/'replay.npz');t=baseline['t'];inputs=baseline['inputs'];source=re.search("source=r'''(.*?)'''",(original/'replay.py').read_text(),re.S).group(1)
results={};arrays={}
for name,corepath in [('v1',original/'controller_v1_source'),('v2',ROOT/'px4_trans/px4/src/modules/nn_control')]:
 with tempfile.TemporaryDirectory(prefix='frozen-replay-') as work:
  work=Path(work);(work/'replay.cpp').write_text(source);subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC','-I',str(corepath),str(work/'replay.cpp'),'-o',str(work/'replay.so')],check=True)
  lib=ctypes.CDLL(str(work/'replay.so'));fp=ctypes.POINTER(ctypes.c_float);lib.tick.argtypes=[fp,fp];out=np.zeros((len(t),13),dtype=np.float32)
  for i in range(len(t)):lib.tick(inputs[i].ctypes.data_as(fp),out[i].ctypes.data_as(fp))
  arrays[name]=out
  k=(t>=18)&(t<27.5);late=(t>=27)&(t<27.5);integral=out[:,7:10]*np.array([.0997526571,.0997526571,.0628326535])
  results[name]={'max_abs_integral_moment_xyz_n_m':abs(integral[k]).max(0).tolist(),'late_grid_common_mean':float(out[late,2:5].mean()),'late_grid_common_min':float(out[late,2:5].mean(1).min()),'late_grid_common_max':float(out[late,2:5].mean(1).max()),'late_target_yaw_deg_mean':float(np.rad2deg(out[late,11]).mean())}
assert np.max(abs(arrays['v2'][:,7:9])* .0997526571)<=.004501
assert abs(results['v2']['late_grid_common_mean'])<.01
np.savez_compressed(OUT/'frozen_replay.npz',t=t,inputs=inputs,**arrays)
(OUT/'frozen_replay_metrics.json').write_text(json.dumps({'scope':'Same recorded inputs, two controller versions; not closed-loop flight validation','results':results},indent=2))
print(json.dumps(results,indent=2))
