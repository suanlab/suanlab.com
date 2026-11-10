"""Independent canonical finite-spectrum oracle with exact Fraction inputs.

Uses a separate high-precision microstate sum, not production binary arithmetic.
Extremely large gaps retain log-probability observations without exponentiating
an unbounded argument. References classify projection before serializing.
"""
import argparse,json,math,random
from pathlib import Path
from fractions import Fraction as F
import mpmath as mp
parser=argparse.ArgumentParser()
parser.add_argument('--output',default='scripts/fixtures/canonical-spectrum-independent.json')
out=Path(parser.parse_args().output)
if out.exists(): raise SystemExit('Refusing to overwrite reference')
kB=1.380649e-23
rng=random.Random(26110661)
minimum=math.ulp(0.)
maximum=float.fromhex('0x1.fffffffffffffp+1023')
def number(f):return mp.mpf(f.numerator)/f.denominator if isinstance(f,F) else mp.mpf(f)
def project(v):
 try:r=float(v)
 except (OverflowError,ValueError):return None
 return None if not math.isfinite(r) or (r==0 and v!=0) else r
def observation(v,log=None):
 return dict(value=project(v),sign=int(mp.sign(v)),logMagnitude=project(mp.log(abs(v)) if log is None and v else log) if v else None)
def case(energies,T,digits):
 with mp.workdps(digits):
  thermal=F(kB)*F(T)
  gaps=[F(e)-F(min(energies)) for e in energies]
  a=[number(g/thermal) for g in gaps]
  g=sum(x==0 for x in gaps)
  excited=[x for x in a if x>0]
  # At a_min>1e6 the finite-state tail corrections are below the retained
  # significant digits, but their log and sign are still explicitly present.
  if excited and min(excited)>10**6:
   amin=min(excited)
   logtail=-amin+mp.log(sum(mp.exp(-(x-amin)) if x-amin<10**6 else 0 for x in excited))
   L=None if g==1 else mp.log(g)
   Lobs=dict(value=None,sign=1,logMagnitude=project(logtail)) if L is None else observation(L)
  else:
   tail=mp.fsum(mp.exp(-x) for x in excited)
   L=mp.log(g)+mp.log1p(tail/g)
   Lobs=observation(L)
  rows=[]
  for gap,x in zip(gaps,a):
   if L is None and gap==0:
    lp=dict(value=None,sign=-1,logMagnitude=project(logtail))
    probability=dict(value=1.,sign=1,logMagnitude=None)
   else:
    l=-x-(L if L is not None else 0)
    lp=observation(l)
    if l < -10**6:probability=dict(value=None,sign=1,logMagnitude=project(l))
    else:probability=dict(value=project(mp.exp(l)),sign=1,logMagnitude=project(l))
   exact=gap/thermal
   rows.append(dict(groundState=gap==0,energyGap=observation(number(gap)),scaledEnergyGap=observation(x),logProbability=lp,probability=probability,
                    exactScaledEnergyGap=dict(numerator=str(exact.numerator),denominator=str(exact.denominator))))
  return dict(energies=energies,temperature=T,digits=digits,groundCount=g,thermalEnergy=observation(number(thermal)),centeredLogPartition=Lobs,states=rows)
ordinary=[case([0.,kB*300*math.log(3)],T,160) for T in [150.,300.,600.]]
for count in [1,2,3,4,8,16,32,2,3,5,7,9,12,20,24,32,4,8,3,5,6]:
 energies=[rng.uniform(-5,5)*kB*300 for _ in range(count)]
 ordinary.append(case(energies,rng.uniform(100,1200),160))
limits=[]
for energies,T in [([0.],minimum),([maximum]*3,minimum),([-maximum,maximum],minimum),
 ([0.,minimum],maximum),([0.,maximum],maximum),([-maximum,maximum],maximum),
 ([0.,1.],minimum),([0.,0.,1.],minimum),([0.,1.,1.],minimum),
 ([0.,kB*300*744],300.),([0.,kB*300*746],300.),([0.,kB*300*1000],300.),
 ([0.,0.,kB*300*1000],300.),([minimum,2*minimum,3*minimum],minimum),
 ([maximum,math.nextafter(maximum,0.)],300.),([0.]*32,maximum),
 ([0.,minimum],minimum),([-minimum,minimum],maximum),
 ([0.,1.,math.nextafter(1.,math.inf)],minimum),([0.,kB*300*36],300.)]:
 limits.append(case(energies,T,1000))
assert len(ordinary)==24 and len(limits)==20
out.write_text(json.dumps(dict(method='Exact Fraction centered logits and independent mpmath microstate sums; controlled negligible-tail asymptotic when minimum scaled gap exceeds 1e6.',ordinary=ordinary,limits=limits),indent=2)+'\n')
print('Wrote 24 ordinary160-digit and 20 limit1000-digit cases')
