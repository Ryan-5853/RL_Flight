"""Build bench payload explicitly, then restore the selected v2 header."""
from pathlib import Path
import subprocess,hashlib,shutil,json
ROOT=Path(__file__).resolve().parents[2]
active=ROOT/'px4_trans/px4/src/modules/nn_control/LqiCoaxialCollectivePayload.hpp'
candidate=ROOT/'px4_trans/generated/airframe-1/yaw-balanced-v3/LqiCoaxialCollectivePayload.hpp'
out=ROOT/'px4_trans/generated/firmware/roll_pitch_damping_v3_20261105';out.mkdir(exist_ok=True)
original=active.read_bytes();old_hash=hashlib.sha256(original).hexdigest()
try:
 active.write_bytes(candidate.read_bytes())
 subprocess.run(['make','micoair_h743-v2_nncontrol'],cwd=ROOT/'px4_trans/px4',check=True)
 shutil.copyfile(ROOT/'px4_trans/px4/build/micoair_h743-v2_nncontrol/micoair_h743-v2_nncontrol.px4',out/'micoair_h743-v2_roll_pitch_damping_v3_BENCH_ONLY.px4')
finally:
 active.write_bytes(original)
assert hashlib.sha256(active.read_bytes()).hexdigest()==old_hash
(out/'selection.json').write_text(json.dumps({'default_profile_changed':False,'active_header_restored':True,'active_header_sha256':old_hash,'bench_payload_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),'flashed':False},indent=2))
print('Bench image archived; selected v2 payload restored; no flashing.')
