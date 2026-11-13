"""Identical recorded inputs: controller-output sensitivity, NOT alternate flight."""
from pathlib import Path
import numpy as np,json,re,ctypes,tempfile,subprocess
from scipy.signal import welch
from scipy.integrate import trapezoid
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
logs=ROOT/'analysis/roll_20261105_unknown1';a=np.load(logs/'replay.npz');t=a['t'];inputs=a['inputs'];source=re.search("source=r'''(.*?)'''",(logs/'replay.py').read_text(),re.S).group(1)
arrays={};results={}
for version in ['v2','v3']:
 header=ROOT/f'px4_trans/generated/airframe-1/yaw-balanced-{version}/LqiCoaxialCollectivePayload.hpp'
 text=source.replace('#include "LqiCoaxialCollectivePayload.hpp"',f'#include "{header}"')
 with tempfile.TemporaryDirectory(prefix='v3-replay-') as work:
  work=Path(work);(work/'core.cpp').write_text(text);subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC','-I',str(ROOT/'px4_trans/px4/src/modules/nn_control'),str(work/'core.cpp'),'-o',str(work/'core.so')],check=True)
  lib=ctypes.CDLL(str(work/'core.so'));fp=ctypes.POINTER(ctypes.c_float);lib.tick.argtypes=[fp,fp];out=np.zeros((len(t),13),dtype=np.float32)
  for i in range(len(t)):lib.tick(inputs[i].ctypes.data_as(fp),out[i].ctypes.data_as(fp))
 k=(t>=19)&(t<32);f,p=welch(out[k,2],fs=500,nperseg=2048);band=(f>=4)&(f<=8);power=float(trapezoid(p[band],f[band]));arrays[version]=out
 results[version]={'servo1_4_8hz_rms':float(np.sqrt(power)),'max_abs_roll_integral_moment':float(abs(out[k,7]*(.0997526571 if version=='v2' else .020)).max()),'motor_command_max':float(out[k,:2].max()),'servo_command_max_abs':float(abs(out[k,2:5]).max())}
ratio=results['v3']['servo1_4_8hz_rms']/results['v2']['servo1_4_8hz_rms']
assert ratio<.6
np.savez_compressed(OUT/'replay_outputs.npz',t=t,**arrays)
(OUT/'replay_metrics.json').write_text(json.dumps({'scope':__doc__,'results':results,'v3_v2_band_rms_ratio':ratio},indent=2));print(json.dumps(results,indent=2));print('4-8Hz recorded-input output RMS ratio',ratio)
