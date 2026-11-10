"""High precision independent references for rational log/negative exp bounds.

Stores conservative integer envelopes around mpmath results at the requested
fixed-point scale. Extra precision makes these envelopes much tighter than one
output quantum. Extreme exponentials below 2^(-P-128) use the rigorous broad
reference [0,1/2^128] in scaled units, and are explicitly marked as such.
"""
import argparse,json,math,random
from pathlib import Path
from fractions import Fraction as F
import mpmath as mp
parser=argparse.ArgumentParser()
parser.add_argument('--output',default='scripts/fixtures/interval-transcendentals-independent.json')
out=Path(parser.parse_args().output)
if out.exists():raise SystemExit('Refusing to overwrite reference')
maximum=float.fromhex('0x1.fffffffffffffp+1023');minimum=math.ulp(0.)
rng=random.Random(26110691)
inputs=[([1.],[1.]),([2.],[1.]),([1.],[2.]),([1.+math.ulp(1.)],[1.]),
 ([maximum],[minimum]),([minimum],[maximum]),([minimum],[1.]),
 ([maximum,maximum],[minimum,minimum]),([minimum,minimum],[maximum,maximum]),
 ([1.],[1.380649e-23,minimum]),([minimum],[1.380649e-23,maximum]),
 ([0.],[1.]),([1.],[3.]),([3.],[1.]),([1.],[8.]),([1.],[16.]),
 ([32.],[1.]),([744.],[1.]),([1000.],[1.]),([10000.],[1.])]
for _ in range(12):inputs.append(([2**rng.uniform(-1000,1000)],[2**rng.uniform(-1000,1000)]))
rows=[]
for i,(nf,df) in enumerate(inputs):
 n=math.prod(map(F,nf));d=math.prod(map(F,df));q=n/d
 for P in ([64,256,1024] if i<20 else [256]):
  with mp.workprec(max(512,P+512)):
   value=mp.mpf(q.numerator)/q.denominator
   # Quantize far more finely than the production endpoint quantum.
   Q=2**128;S=mp.mpf(2)**P
   def envelope(v):
    scaled=v*S*Q
    # Conservative guards around the high precision transcendental oracle;
    # ±2 fine quanta dominate the retained rounding error in these references.
    return dict(lower=str(int(mp.floor(scaled))-2),upper=str(int(mp.ceil(scaled))+2))
   log=None if q==0 else dict(lower='0',upper='0') if q==1 else envelope(mp.log(value))
   if q==0:
    exp=dict(lower=str(2**(P+128)),upper=str(2**(P+128)));method='exact-zero-argument'
   elif value>(P+128)*mp.log(2):
    exp=dict(lower='0',upper='1');method='analytic-small-exponential-bound'
   else:
    exp=envelope(mp.exp(-value));method='mpmath-direct'
    # Intersect the guarded numerical envelope with 0<exp(-a)<=1.
    exp['lower']=str(max(0,int(exp['lower'])))
    exp['upper']=str(min(2**(P+128),int(exp['upper'])))
   rows.append(dict(numeratorFactors=nf,denominatorFactors=df,precisionBits=P,fineScaleBits=128,
                    logarithm=log,negativeExponential=exp,exponentialReferenceMethod=method))
assert len(rows)==72
out.write_text(json.dumps(dict(method='Independent mpmath at P+512 bits (minimum512), guarded fine-grid reference envelopes; explicit analytic bound for tiny exponentials.',cases=rows),indent=2)+'\n')
print('Wrote72 references at64,256,1024 precision bits')
