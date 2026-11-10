"""Independent physical-coordinate work, heat and reversible entropy integrals.

Binary input path coordinates are reconstructed with fractions.Fraction. No
production imports, atanh expansion or integer log bound implementation is used.
"""
import argparse
import json
import math
import random
from fractions import Fraction as F
from pathlib import Path
import mpmath as mp

parser=argparse.ArgumentParser()
parser.add_argument('--output',default='scripts/fixtures/ideal-gas-path-independent.json')
output=Path(parser.parse_args().output)
if output.exists():raise SystemExit('Refusing to overwrite an existing reference')
rng=random.Random(26110618)
Rfloat=8.31446261815324
base=dict(moles=1.,initialTemperature=300.,finalTemperature=330.,initialVolume=.01,finalVolume=.02)
def observation(v):return dict(value=str(v),sign=int(mp.sign(v)),logMagnitude=None if v==0 else str(mp.log(abs(v))))
def value(f):return mp.mpf(f.numerator)/f.denominator

def case(p,path,progress,digits,integrals=False):
 with mp.workdps(digits):
  n,R=value(F(p['moles'])),value(F(Rfloat));Cv=mp.mpf('1.5')*R
  t0,t1,v0,v1=[F(p[k]) for k in ['initialTemperature','finalTemperature','initialVolume','finalVolume']]
  s=F(progress);first=min(2*s,F(1));second=max(2*s-1,F(0))
  rt,rv=(first,second) if path=='A' else (second,first)
  t=t0+(t1-t0)*rt;v=v0+(v1-v0)*rv
  T,V,T0,V0=map(value,[t,v,t0,v0]);iso=value(t1 if path=='A' else t0)
  # Integrating p(V) over the actual isothermal section and Cv dT over
  # the actual isochore computes the path budget independently of production.
  if integrals:
   W=mp.quad(lambda x:n*R*iso/x,[V0,V]);U=mp.quad(lambda x:n*Cv,[T0,T])
   S=mp.quad(lambda x:n*Cv/x,[T0,T])+mp.quad(lambda x:n*R/x,[V0,V])
  else:
   W=n*R*iso*mp.log(V/V0);U=n*Cv*(T-T0)
   S=n*Cv*mp.log(T/T0)+n*R*mp.log(V/V0)
  # Recognize algebraically exact state entropy zeros; do not promote the
  # numerical residue of two high-precision logarithms to a physical value.
  if t**3*v**2==t0**3*v0**2:S=mp.mpf(0)
  expected=dict(temperature=T,volume=V,pressure=n*R*T/V,temperatureChange=T-T0,volumeChange=V-V0,
                energyChange=U,work=W,heat=U+W,entropyChange=S)
  row=dict(parameters=p,path=path,progress=progress,digits=digits,physicalCoordinateIntegrals=integrals,
           expected={k:observation(v) for k,v in expected.items()})
  if integrals:
   assert mp.almosteq(W,n*R*iso*mp.log(V/V0),rel_eps=mp.mpf('1e-100'),abs_eps=mp.mpf('1e-100'))
   assert mp.almosteq(U,n*Cv*(T-T0),rel_eps=mp.mpf('1e-100'),abs_eps=mp.mpf('1e-100'))
  return row
ordinary=[]
for i in range(24):
 p=base.copy() if i==0 else dict(moles=rng.uniform(.25,2),initialTemperature=rng.uniform(250,350),
   finalTemperature=rng.uniform(250,350),initialVolume=rng.uniform(.005,.04),finalVolume=rng.uniform(.005,.04))
 path='A' if i%2==0 else 'B';s=[0.,.125,.25,.5,.75,1.][i%6] if i<12 else rng.random()
 ordinary.append(case(p,path,s,160,integrals=True));print(f'ordinary {i+1}/24',flush=True)
small=math.nextafter(0.,1.)
adiabat=dict(base,initialTemperature=252.,finalTemperature=343.,initialVolume=343/32768,finalVolume=216/32768)
limits=[
 (base,'A',small),(base,'B',small),
 (dict(base,finalVolume=math.nextafter(.01,math.inf)),'B',small),
 (dict(base,finalTemperature=math.nextafter(300.,math.inf)),'A',small),
 (dict(base,finalTemperature=300.,finalVolume=.01),'A',1.),
 (dict(base,finalTemperature=300.,finalVolume=.01),'B',1.),
 (adiabat,'A',1.),(adiabat,'B',1.),
 (dict(adiabat,finalVolume=math.nextafter(adiabat['finalVolume'],math.inf)),'A',1.),
 (dict(adiabat,finalVolume=math.nextafter(adiabat['finalVolume'],0.)),'B',1.),
 (dict(base,initialTemperature=350.,finalTemperature=250.,initialVolume=.005,finalVolume=.04),'A',1.),
 (dict(base,initialTemperature=250.,finalTemperature=350.,initialVolume=.04,finalVolume=.005),'B',1.),
 (base,'A',math.nextafter(.5,1.)),(base,'B',math.nextafter(.5,1.)),
]
# Nearby binary volume endpoints lie on both sides of a heat cancellation.
for path,iso in [('A',300.),('B',330.)]:
 root=.015625*math.exp(45/iso)
 for volume in [math.nextafter(root,0.),root,math.nextafter(root,math.inf)]:
  limits.append((dict(base,initialTemperature=330.,finalTemperature=300.,initialVolume=.015625,finalVolume=volume),path,1.))
# Rounded energy/work addition is zero here; actual path heat is positive.
limits.append((dict(base,initialTemperature=251.,finalTemperature=271.,initialVolume=.015625,finalVolume=.013987598122283916),'A',1.))
edge=[]
for i,(p,path,s) in enumerate(limits):
 edge.append(case(p,path,s,1000));print(f'limit {i+1}/{len(limits)}',flush=True)
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps(dict(method='exact binary path coordinates and physical p dV, Cv dT, deltaQrev/T integrals',gasConstantBinary=Rfloat,ordinary=ordinary,limits=edge),indent=2)+'\n')
print(f'Saved24 ordinary and{len(edge)} limit references to {output}',flush=True)
