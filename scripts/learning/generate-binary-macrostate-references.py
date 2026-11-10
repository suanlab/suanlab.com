"""Independent factorial counts, explicit small-state enumeration and mpmath.

Production uses binomial-row recurrence and summed conditional moments. This
oracle uses factorial quotients and closed finite-population moments, with
exhaustive bit-string counting for N<=12.
"""
import argparse,json,math,random
from pathlib import Path
from fractions import Fraction as F
import mpmath as mp
parser=argparse.ArgumentParser()
parser.add_argument('--output',default='scripts/fixtures/binary-macrostates-independent.json')
out=Path(parser.parse_args().output)
if out.exists():raise SystemExit('Refusing to overwrite reference')
rng=random.Random(26110711);kB=1.380649e-23;minimum=math.ulp(0.);maximum=float.fromhex('0x1.fffffffffffffp+1023')
def choose(n,k):return math.factorial(n)//(math.factorial(k)*math.factorial(n-k))
def number(v):return mp.mpf(v.numerator)/v.denominator if isinstance(v,F) else mp.mpf(v)
def obs(v):
 v=number(v)
 try:r=float(v)
 except OverflowError:r=None
 if v and (r==0 or r is None or not math.isfinite(r)):r=None
 return dict(value=r,sign=int(mp.sign(v)),logMagnitude=None if not v else float(mp.log(abs(v))))
def case(n,k,a,gap):
 with mp.workdps(160):
  counts=[choose(n,j) for j in range(n+1)];total=2**n
  if n<=12:
   enumerated=[0]*(n+1)
   for mask in range(2**n):enumerated[mask.bit_count()]+=1
   assert enumerated==counts
  H=-mp.fsum(number(F(c,total))*mp.log(number(F(c,total))) for c in counts)
  conditional=mp.fsum(number(F(c,total))*mp.log(c) for c in counts)
  lo=max(0,k-(n-a));hi=min(k,a)
  block=[(j,choose(a,j)*choose(n-a,k-j)) for j in range(lo,hi+1)]
  assert sum(c for _,c in block)==counts[k]
  mean=F(a*k,n) if n else F(0)
  variance=F(a*k*(n-k)*(n-a),n*n*(n-1)) if n>1 else F(0)
  return dict(sites=n,excitations=k,blockASites=a,energyGap=gap,enumerated=n<=12,
   totalMicrostates=str(total),multiplicity=str(counts[k]),macroModes=[j for j,c in enumerate(counts) if c==max(counts)],
   selectedProbability=obs(F(counts[k],total)),withinShellProbability=obs(F(1,counts[k])),logMultiplicity=obs(mp.log(counts[k])),
   selectedMacroEntropyContribution=obs(-number(F(counts[k],total))*mp.log(number(F(counts[k],total)))),
   boltzmannEntropy=obs(number(F(kB))*mp.log(counts[k])),selectedEnergy=obs(F(k)*F(gap)),
   fullSpaceInformationEntropy=obs(n*mp.log(2)),macroInformationEntropy=obs(H),meanWithinShellInformationEntropy=obs(conditional),
   conditionalBlockMean=obs(mean),conditionalBlockVariance=obs(variance),
   conditionalBlockModes=[j for j,c in block if c==max(x[1] for x in block)],
   blockProbes=[dict(excitationsA=j,count=str(c),probability=obs(F(c,counts[k]))) for i,(j,c) in enumerate(block) if i in {0,len(block)//2,len(block)-1}])
inputs=[(0,0,0,1.),(1,0,0,1.),(1,1,1,1.),(4,2,2,kB*300),(8,4,4,kB*300),(12,3,5,kB*300)]
for _ in range(18):
 n=rng.randint(2,80);inputs.append((n,rng.randint(0,n),rng.randint(0,n),10**rng.uniform(-100,100)))
ordinary=[case(*v) for v in inputs]
limits=[case(*v) for v in [(256,128,128,minimum),(1024,0,512,1.),(1075,0,512,1.),(1080,0,512,1.),
 (4096,0,2048,maximum),(4096,4096,2048,minimum),(4096,2048,2048,kB*300),
 (4095,2047,2047,kB*300),(4096,1,1,maximum),(4096,4095,4095,minimum),
 (64,32,0,1.),(64,32,64,1.)]]
out.write_text(json.dumps(dict(method='Independent factorial multiplicities; exhaustive bit strings for N<=12; direct mpmath entropy sums and closed finite-population moments.',ordinary=ordinary,limits=limits),indent=2)+'\n')
print('Wrote24 ordinary and12 limiting160-digit cases')
