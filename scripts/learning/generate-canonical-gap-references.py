"""Independent microstate KL oracle, not the production interval identity."""
import argparse,json,math,random
from pathlib import Path
from fractions import Fraction as F
import mpmath as mp
parser=argparse.ArgumentParser()
parser.add_argument('--output',default='scripts/fixtures/canonical-gap-independent.json')
out=Path(parser.parse_args().output)
if out.exists():raise SystemExit('Refusing to overwrite reference')
kB=1.380649e-23;minimum=math.ulp(0.);maximum=float.fromhex('0x1.fffffffffffffp+1023')
rng=random.Random(26110701)
def number(v):return mp.mpf(v.numerator)/v.denominator if isinstance(v,F) else mp.mpf(v)
def observation(v):
 try:value=float(v)
 except OverflowError:value=None
 if v and (value==0 or value is None or not math.isfinite(value)):value=None
 return dict(value=value,sign=int(mp.sign(v)),logMagnitude=None if not v else float(mp.log(abs(v))))
def case(E,T,w,copies,digits):
 with mp.workdps(digits):
  thermal=F(kB)*F(T);a=[number((F(e)-F(min(E)))/thermal) for e in E]
  p=[F(x)/sum(map(F,w)) for x in w]
  # Direct high precision target and per-microstate convex remainder.
  target=[mp.exp(-x) for x in a];Z=mp.fsum(target);q=[x/Z for x in target]
  terms=[]
  for pi,qi in zip(p,q):
   pp=number(pi)
   term=qi if not pi else pp*mp.log(pp/qi)-pp+qi
   assert term>=0
   terms.append(term)
  D=mp.fsum(terms);gap=D*number(thermal)
  referenceBits=(256 if digits==160 else 8192)+128
  scaled=D*mp.mpf(2)**referenceBits
  envelope=dict(lower='0',upper='0') if D==0 else dict(lower=str(int(mp.floor(scaled))-2),upper=str(int(mp.ceil(scaled))+2))
  return dict(energies=E,temperature=T,trialWeights=w,copies=copies,digits=digits,
              referenceBits=referenceBits,divergenceEnvelope=envelope,
              divergence=observation(D),perCopyGap=observation(gap),totalGap=observation(gap*copies))
ordinary=[case([0.,kB*300*math.log(3)],300.,w,16.,160) for w in [[1.,1.],[3.,1.],[1.,0.]]]
for n in [1,2,3,4,8,16,32,2,3,5,7,9,12,20,24,32,4,8,3,5,6]:
 E=[rng.uniform(-5,5)*kB*300 for _ in range(n)]
 w=[10**rng.uniform(-5,5) if rng.random()>.2 else 0. for _ in range(n)]
 if not any(w):w[0]=1.
 ordinary.append(case(E,rng.uniform(100,1200),w,rng.randint(1,1000000),160))
limits=[]
for E,T,w,copies in [([0.,0.],300.,[1.,1.],1.),([0.,0.],minimum,[1.,2.],maximum),
 ([0.,minimum],maximum,[1.,1.],maximum),([0.,minimum],minimum,[1.,1.],maximum),
 ([0.,1.],minimum,[1.,1.],1.),([0.,0.],maximum,[maximum,minimum],maximum),
 ([0.,kB*300*math.log(3)],300.,[3.,math.nextafter(1.,2.)],maximum),
 ([0.,0.,0.],300.,[1.,1.,math.nextafter(1.,2.)],1.),
 ([0.,0.,1.],300.,[1.,2.,0.],1.),([0.]*32,maximum,[minimum]*32,maximum)]:
 limits.append(case(E,T,w,copies,3000))
assert len(ordinary)==24 and len(limits)==10
out.write_text(json.dumps(dict(method='Independent mpmath normalized microstate exponentials and direct convex KL contributions; Fraction trial normalization.',ordinary=ordinary,limits=limits),indent=2)+'\n')
print('Wrote24 ordinary160-digit and10 limit3000-digit direct microstate KL references')
