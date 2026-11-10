"""Independent Feq references with exact anchors and mpmath partition sums.

Very small corrections are retained logarithmically. Directed rounding at exact
anchor midpoints is checked using Fraction and nextafter, without a sentinel.
"""
import argparse,json,math,random
from fractions import Fraction as F
from pathlib import Path
import mpmath as mp
parser=argparse.ArgumentParser()
parser.add_argument('--output',default='scripts/fixtures/equilibrium-free-energy-independent.json')
out=Path(parser.parse_args().output)
if out.exists():raise SystemExit('Refusing to overwrite reference')
kB=1.380649e-23
minimum=math.ulp(0.)
maximum=float.fromhex('0x1.fffffffffffffp+1023')
rng=random.Random(26110681)
def number(v):return mp.mpf(v.numerator)/v.denominator if isinstance(v,F) else mp.mpf(v)
def projected(v):
 try:r=float(v)
 except OverflowError:return None
 return None if not math.isfinite(r) or (r==0 and v!=0) else r
def observation(v,log=None):
 v=number(v)
 return dict(value=projected(v),sign=int(mp.sign(v)),logMagnitude=None if not v else projected(mp.log(abs(v)) if log is None else log))
def below_anchor(anchor):
 try:f=float(anchor)
 except OverflowError:
  return maximum if anchor==F(maximum)+F(2)**970 else None
 if not math.isfinite(f):return None
 if F(f)>anchor:
  lower=math.nextafter(f,-math.inf)
  if math.isfinite(lower) and (F(f)+F(lower))/2==anchor:f=lower
 return f

def case(energies,T,copies,digits):
 with mp.workdps(digits):
  thermal=F(kB)*F(T);Emin=F(min(energies))
  a=[number((F(e)-Emin)/thermal) for e in energies]
  g=sum(x==0 for x in a);excited=[x for x in a if x>0]
  amin=min(excited) if excited else None
  # For large but manageable a use log1p of the direct microstate tail. For
  # unbounded a the first-order log-tail has an exponentially small error.
  if amin is not None and amin>10000 and g==1:
   logL=-amin+mp.log(mp.fsum(mp.exp(-(x-amin)) if x-amin<10000 else 0 for x in excited))
   infinitesimal=True;L=None
  else:
   tail=mp.fsum(mp.exp(-x) if x<10000 else 0 for x in excited)
   L=mp.log(g)+mp.log1p(tail/g)
   logL=mp.log(L) if L else None
   infinitesimal=False
  groups={}
  for name,count in [('perCopy',1),('total',copies)]:
   anchor=Emin*F(count)
   if count==0 or (L is not None and L==0):
    corr=observation(0);free=observation(anchor)
   elif infinitesimal:
    logC=mp.log(number(thermal))+logL+mp.log(count)
    corr=dict(value=None,sign=1,logMagnitude=projected(logC))
    free=dict(value=None,sign=-1,logMagnitude=projected(logC)) if anchor==0 else dict(value=below_anchor(anchor),sign=1 if anchor>0 else -1,logMagnitude=projected(mp.log(abs(number(anchor)))))
   else:
    C=number(thermal)*L*count
    corr=observation(C);free=observation(number(anchor)-C)
   groups[name]=dict(helmholtz=free,centeredPartitionEnergyCorrection=corr)
  return dict(energies=energies,temperature=T,copies=copies,digits=digits,**groups)
ordinary=[case([0.,kB*300*math.log(3)],T,16.,160) for T in [150.,300.,600.]]
for n in [1,2,3,4,8,16,32,2,3,5,7,9,12,20,24,32,4,8,3,5,6]:
 E=[rng.uniform(-5,5)*kB*300 for _ in range(n)]
 ordinary.append(case(E,rng.uniform(100,1200),rng.randint(0,1000000),160))
inputs=[([0.],minimum,maximum),([maximum],300.,maximum),([-maximum],300.,maximum),
 ([0.,1.],minimum,maximum),([1.,2.],minimum,3.),([-1.,0.],minimum,3.),
 ([0.,0.],minimum,maximum),([0.,0.],maximum,maximum),
 ([0.,kB*maximum*744],maximum,1.),([0.,kB*maximum*746],maximum,1.),
 ([0.,kB*maximum*1000],maximum,1.),([0.,kB*300*1000],300.,maximum),
 ([0.,kB*300*1500],300.,maximum),([0.,kB*300*10000],300.,maximum),
 ([0.,kB*300*20000],300.,maximum),([0.,maximum],minimum,1.),
 ([-maximum,maximum],minimum,maximum),([minimum,2*minimum],minimum,maximum),
 ([0.]*32,maximum,maximum),([0.,0.,1.],minimum,maximum),
 ([math.nextafter(1.,2.),2.],minimum,3.),
 ([-math.nextafter(1.,2.),0.],minimum,3.),
 ([0.,minimum],maximum,1.),([0.,maximum],maximum,0.)]
overflow_tie_anchor=float(F(2**54-1,3)*F(2)**970)
inputs.extend([([overflow_tie_anchor,maximum],minimum,3.),
               ([-overflow_tie_anchor,0.],minimum,3.),
               ([-(1.+3*math.ulp(1.)),0.],minimum,3.)])
limits=[case(*x,1000) for x in inputs]
assert len(ordinary)==24 and len(limits)==27
out.write_text(json.dumps(dict(method='Fraction anchors and independent mpmath microstate partition sums; for extremely small corrections, logarithmic asymptotics plus exact one-sided Fraction midpoint rounding.',ordinary=ordinary,limits=limits),indent=2)+'\n')
print('Wrote 24 ordinary160-digit and 27 limit1000-digit cases')
