"""Independent trial-functionals oracle: exact input moments and mpmath entropy."""
import argparse,json,math,random
from fractions import Fraction as F
from pathlib import Path
import mpmath as mp
parser=argparse.ArgumentParser()
parser.add_argument('--output',default='scripts/fixtures/trial-thermodynamics-independent.json')
out=Path(parser.parse_args().output)
if out.exists():raise SystemExit('Refusing to overwrite reference')
kB=1.380649e-23
minimum=math.ulp(0.)
maximum=float.fromhex('0x1.fffffffffffffp+1023')
rng=random.Random(26110671)
def number(v):return mp.mpf(v.numerator)/v.denominator if isinstance(v,F) else mp.mpf(v)
def observation(v):
 v=number(v)
 try:projected=float(v)
 except OverflowError:projected=None
 if v and (projected==0 or projected is None or not math.isfinite(projected)):projected=None
 return dict(value=projected,sign=int(mp.sign(v)),logMagnitude=None if v==0 else float(mp.log(abs(v))))
def case(energies,T,w,copies,digits):
 with mp.workdps(digits):
  wf=list(map(F,w));p=[x/sum(wf) for x in wf]
  e=[pi*F(E) for pi,E in zip(p,energies)]
  h=[]
  for pi in p:
   lp=0 if pi==0 else mp.log1p(number(pi-1)) if pi>=F(1,2) else mp.log(number(pi))
   h.append(-number(pi)*lp)
  U=sum(e);H=mp.fsum(h);S=number(F(kB))*H;TS=number(F(kB)*F(T))*H
  per=dict(meanEnergy=number(U),informationEntropy=H,entropy=S,temperatureTimesEntropy=TS,helmholtz=number(U)-TS)
  return dict(energies=energies,temperature=T,trialWeights=w,copies=copies,digits=digits,
   perCopy={k:observation(v) for k,v in per.items()},total={k:observation(v*number(F(copies))) for k,v in per.items()},
   states=[dict(probability=observation(pi),meanEnergyContribution=observation(ei),informationEntropyContribution=observation(hi)) for pi,ei,hi in zip(p,e,h)])
ordinary=[case([0.,kB*300*math.log(3)],300.,w,16.,160) for w in [[1.,1.],[3.,1.],[1.,0.]]]
for n in [1,2,3,4,8,16,32,2,3,5,7,9,12,20,24,32,4,8,3,5,6]:
 E=[rng.uniform(-5,5)*kB*300 for _ in range(n)]
 w=[10**rng.uniform(-5,5) if rng.random()>.2 else 0. for _ in range(n)]
 if not any(w):w[0]=1.
 ordinary.append(case(E,rng.uniform(100,1200),w,rng.randint(0,1000000),160))
limit_inputs=[
 ([0.],minimum,[minimum],maximum),([0.,0.],minimum,[1.,1.],maximum),
 ([maximum,-maximum],300.,[maximum,maximum],1.),
 ([maximum,maximum],maximum,[maximum,maximum],maximum),
 ([0.,0.],maximum,[maximum,minimum],maximum),
 ([0.,minimum],maximum,[1.,minimum],maximum),
 ([minimum,-minimum],minimum,[1.,1.],maximum),
 ([0.,1.],minimum,[1.,minimum],maximum),
 ([0.,1.],maximum,[maximum,minimum],maximum),
 ([0.,1.],300.,[1.,0.],maximum),
 ([1.,0.],300.,[1.,0.],0.),
 ([kB*256*math.log(2)]*2,256.,[1.,1.],1.),
 ([0.,0.],maximum,[1.,minimum],1.),
 ([maximum,-maximum],minimum,[1.,1.],1.),
 ([1.,-1.],300.,[maximum,math.nextafter(maximum,0.)],1.),
 ([minimum,minimum],minimum,[1.,1.],maximum),
 ([0.]*32,maximum,[minimum]*32,maximum),
 ([-maximum,maximum],300.,[minimum,maximum],1.),
 ([0.,0.],300.,[minimum,maximum],maximum),
 ([0.,1.,2.],minimum,[0.,minimum,minimum],maximum)]
approx=F(2)*F(kB)*256*F(math.log(2))
high=float(approx)
low=float(approx-F(high))
limit_inputs.append(([high,low],256.,[1.,1.],1.))
limits=[case(*args,1000) for args in limit_inputs]
assert len(ordinary)==24 and len(limits)==21
out.write_text(json.dumps(dict(method='Exact Fraction normalization and energy moments; independent mpmath direct information entropy with log1p near probability one.',ordinary=ordinary,limits=limits),indent=2)+'\n')
print('Wrote 24 ordinary160-digit and 21 limit1000-digit cases')
