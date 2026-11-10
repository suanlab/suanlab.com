"""Independent ground-manifold KL from an mpmath microstate partition tail.

Uses log1p(tail/g) directly, without the production log-D expansion enclosure.
Exact Fraction input coordinates precede 1000-digit transcendental evaluation.
"""
import argparse,json,math
from pathlib import Path
from fractions import Fraction as F
import mpmath as mp
parser=argparse.ArgumentParser()
parser.add_argument('--output',default='scripts/fixtures/canonical-ground-tail-independent.json')
out=Path(parser.parse_args().output)
if out.exists():raise SystemExit('Refusing to overwrite reference')
kB=1.380649e-23;minimum=math.ulp(0.);maximum=float.fromhex('0x1.fffffffffffffp+1023')
def number(v):return mp.mpf(v.numerator)/v.denominator if isinstance(v,F) else mp.mpf(v)
def projection(v):
 try:r=float(v)
 except (OverflowError,ValueError):return None
 return None if not math.isfinite(r) or (r==0 and v!=0) else r
def observation(v):return dict(value=projection(v),sign=int(mp.sign(v)),logMagnitude=None if not v else projection(mp.log(abs(v))))
def case(E,T,w,N):
 with mp.workdps(1000):
  thermal=F(kB)*F(T);emin=F(min(E));a=[number((F(e)-emin)/thermal) for e in E]
  g=sum(x==0 for x in a);tail=mp.fsum(mp.exp(-x) for x in a if x>0)
  D=mp.log1p(tail/g);assert D>0
  logD=mp.log(D);gap=D*number(thermal)
  S=mp.mpf(2)**384;scaled=logD*S
  envelope=dict(lower=str(int(mp.floor(scaled))-2),upper=str(int(mp.ceil(scaled))+2))
  return dict(energies=E,temperature=T,trialWeights=w,copies=N,groundCount=g,
   divergence=observation(D),perCopyGap=observation(gap),totalGap=observation(gap*N),logDivergence=observation(logD),
   logReferenceBits=384,logEnvelope=envelope)
inputs=[([0.,1.],minimum,[1.,0.],1.),([0.,maximum],minimum,[maximum,0.],maximum),
 ([0.,0.,1.],minimum,[1.,1.,0.],maximum),([0.,0.,0.,1.,2.],minimum,[1.,1.,1.,0.,0.],maximum),
 ([-maximum,maximum],minimum,[1.,0.],1.),([1.,2.],minimum,[1.,0.],0.),
 ([0.,kB*maximum*1000],maximum,[1.,0.],1.),([0.,kB*300*1000],300.,[1.,0.],maximum),
 ([0.,0.,kB*300*1000],300.,[1.,1.,0.],maximum),
 ([0.,kB*300*10000],300.,[1.,0.],maximum),([0.,kB*300*20000],300.,[1.,0.],maximum),
 ([0.,kB*maximum*744],maximum,[1.,0.],1.),
 ([0.,kB*maximum*746],maximum,[1.,0.],1.),
 ([0.,kB*300*100,kB*300*101],300.,[1.,0.,0.],1.),
 ([0.,0.,kB*300*100,kB*300*100],300.,[1.,1.,0.,0.],1.),
 ([0.]+[1.]*31,minimum,[1.]+[0.]*31,maximum),
 ([0.]*31+[1.],minimum,[minimum]*31+[0.],maximum),
 ([0.,kB*300*1500],300.,[1.,0.],maximum)]
rows=[case(*x) for x in inputs]
out.write_text(json.dumps(dict(method='Fraction energy coordinates and independent mpmath1000-digit direct log1p of the excited microstate tail divided by ground degeneracy.',cases=rows),indent=2)+'\n')
print('Wrote18 independent1000-digit ground-tail references')
