"""Check full steady force/moment composition using the existing unit plant equations.

Reads its exact physical YAML baseline; works without the SimEnv Python package.
"""
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[2]
pair=yaml.safe_load((ROOT/'px4_trans/configs/composite_controller.yaml').read_text())
physical=yaml.safe_load((ROOT/pair['unit_model']).read_text())
def value(node,key): return np.asarray(node[key]['value'])
aero=physical['aerodynamics']
def local_wrench(command):
    speed=[np.interp(command[i],*value(physical['motors'][i],'pwm_to_rpm_table').T) for i in range(2)]
    coeff=value(aero,'thrust_coefficients')
    thrust=coeff@[speed[0]**2,speed[1]**2,speed[0]*speed[1]]
    neutral=value(aero,'neutral_thrust_direction_b')
    partition=np.array([float(v['value']) for v in aero['thrust_partition'].values()])
    force=thrust*partition[0]*neutral
    moment=np.cross(value(aero,'direct_thrust_center_b')-value(physical['body'],'center_of_mass_b'),force)
    angles=np.array([np.interp(command[i+2],*value(physical['servos'][i],'pwm_angle_table').T) for i in range(3)])
    attenuation=np.array([np.interp(abs(angles[i]),*value(g,'self_attenuation_curve').T) for i,g in enumerate(aero['grids'])])
    effective=np.clip(attenuation-value(aero,'coupling_attenuation')@(1-attenuation),0,1)
    for i,g in enumerate(aero['grids']):
        axis=value(g,'deflection_axis_b')
        angle=float(value(g['vector_deflection'],'gain'))*angles[i]+float(value(g['vector_deflection'],'offset'))
        direction=neutral*np.cos(angle)+np.cross(axis,neutral)*np.sin(angle)+axis*(axis@neutral)*(1-np.cos(angle))
        grid_force=thrust*partition[i+1]*effective[i]*direction
        force+=grid_force
        moment+=np.cross(value(g,'aerodynamic_center_b')-value(physical['body'],'center_of_mass_b'),grid_force)
    torque=[float(value(m,'torque_coefficient'))*s*s for m,s in zip(physical['motors'],speed)]
    moment+=(torque[0]-torque[1])*(-neutral)
    return force,moment,thrust

def compose(command):
    force=np.zeros(3); moment=np.zeros(3); lever=np.zeros(3); thrust=[]
    for cmd,angle,y in zip(command,np.radians(pair['installation_yaw_deg']),[pair['arm_m'],-pair['arm_m']]):
        c,s=np.cos(angle),np.sin(angle)
        rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        f,m,t=local_wrench(cmd)
        fc=rotation@f; mc=rotation@m
        arm=np.cross([0,y,0],fc)
        force+=fc; moment+=mc+arm; lever+=arm; thrust.append(t)
    return np.r_[-force[2],moment],lever,np.array(thrust)

with tempfile.TemporaryDirectory() as tmp:
    binary=Path(tmp)/'pair'
    subprocess.run(['g++','-std=c++17','-O2',str(Path(__file__).with_name('pair_control_check.cpp')),'-o',str(binary)],check=True)
    rows=np.array([list(map(float,line.split())) for line in subprocess.check_output([str(binary),'dump'],text=True).splitlines()])
results=[]
for row in rows:
    achieved,lever,thrust=compose(row[5:15].reshape(2,5))
    results.append((achieved,thrust))
    print(f'axis {int(row[0])}: requested={row[1:5]}, model={achieved}, lever={lever}')
    assert np.isfinite(achieved).all()
    np.testing.assert_allclose(row[15:19],achieved,atol=1e-5,rtol=1e-4)
assert abs(results[0][0][0]-rows[0,1])<.02
for axis in range(3): assert results[axis+1][0][axis+1]>0
error=np.max(abs(results[3][1]-results[0][1]))
assert error<.02,error
print(f'Yaw side thrust error: {error:.6g} N; nonlinear pair checks passed')

# An asymmetric grid command must contribute a nonzero lever-arm moment.
command=rows[2,5:15].reshape(2,5).copy()
command[1,2:]*=.3
_,lever,_=compose(command)
assert np.linalg.norm(lever)>1e-6,lever
print('Asymmetric grid lever-arm term:',lever)
