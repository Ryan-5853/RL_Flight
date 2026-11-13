from pathlib import Path
import subprocess,tempfile,json,re,importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
candidate=ROOT/'px4_trans/generated/airframe-1/yaw-balanced-v3'
with tempfile.TemporaryDirectory(prefix='damped-core-tests-') as temp:
 for name in ['lqi_coaxial_collective_core_check','yaw_balance_envelope_check','shared_yaw_antiwindup_check']:
  binary=Path(temp)/name
  subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror',f'-DCOAXIAL_PAYLOAD_HEADER="{candidate / "LqiCoaxialCollectivePayload.hpp"}"',f'-DCOAXIAL_GOLDEN_HEADER="{candidate / "lqi_coaxial_collective_golden.hpp"}"','-I',str(ROOT/'px4_trans/px4/src/modules/nn_control'),str(ROOT/f'px4_trans/tests/{name}.cpp'),'-o',str(binary)],check=True)
  subprocess.run([str(binary)],check=True)
spec=importlib.util.spec_from_file_location('gen',ROOT/'px4_trans/tools/generate_coaxial_collective_backend.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
config=ROOT/'px4_trans/configs/lqi_coaxial_collective_airframe1_yaw_balanced_v3.yaml';mp=candidate/'lqi_coaxial_collective_manifest.json';manifest=json.loads(mp.read_text());model,dep,meta=g.load_reference_update(config,mp)
assert g.checksum(model)==int(manifest['model_checksum'],16)
old_cfg=ROOT/'px4_trans/configs/lqi_coaxial_collective_airframe1_yaw_balanced_v2.yaml';old_mp=ROOT/'px4_trans/generated/airframe-1/yaw-balanced-v2/lqi_coaxial_collective_manifest.json';old,_,_=g.load_reference_update(old_cfg,old_mp)
assert np.array_equal(model['attitude_gain'][2],old['attitude_gain'][2])
for key in ['upper_table','lower_table','maximum_moment','yaw_balance','manual_command_decay']:
 assert np.array_equal(model[key],old[key]),key
for axis in range(2):assert np.allclose(model['attitude_gain'][axis,[axis,2+axis,5+axis]],[.012,.004,.020])
assert model['roll_pitch_integral_moment_fraction']==.9
(OUT/'candidate_checks.json').write_text(json.dumps({'host_core_checks':'3 passed','payload_checksum_roundtrip':True,'yaw_and_plant_tables_unchanged':True,'new_pid':[.012,.004,.020],'integral_fraction':.9,'model_checksum':manifest['model_checksum']},indent=2))
print('candidate checksum, gains, bounds and unchanged yaw/tables passed')
