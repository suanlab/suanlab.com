"""Independent Decimal-CDF quantization and integer seeded-path references.
Uses the separately generated actual-binary64 Gibbs reference, not TypeScript.
An LCG trace is deterministic; Q assumes independent uniform finite-bit draws.
"""
from decimal import Decimal as D,localcontext,ROUND_CEILING
from fractions import Fraction as F
from pathlib import Path
import json
source=json.loads(Path('scripts/fixtures/finite-gibbs-oracle.json').read_text())
byid={c['id']:c for c in source['cases']}
requests=[('near-ideal-ln2','first',8,0,32),('near-ideal-ln2','second',8,0,32),
 ('near-ideal-ln2','random',8,0,64),('near-ideal-ln2','random',8,1,64),
 ('near-ideal-ln2','systematic',8,1,64),('near-ideal-ln2','blocked',8,0,64),
 ('near-ideal-ln2','random',32,2**32-1,64),('fixed-zero-sum-uniform','random',8,1,64),
 ('fixed-zero-sum-uniform','blocked',8,1,64),('fixed-zero-sum-biased','blocked',16,1,64),
 ('rare-conditional','random',8,1,64),('weak-interaction','systematic',8,1,64)]
rows=[]
for caseid,mode,bits,initial_seed,steps in requests:
 case=byid[caseid];p=case['parameters'];r=case['reference'];n=len(r['target']);M=2**bits
 states=[(-1,-1),(-1,1),(1,-1),(1,1)]
 if p['support']=='fixed-zero-sum':states=[s for s in states if sum(s)==0]
 J,h1,h2,c=map(lambda x:F.from_float(float(x)),[p['couplingEnergy'],p['firstFieldEnergy'],p['secondFieldEnergy'],p['energyOffset']])
 energies=[c-J*a*b-h1*a-h2*b for a,b in states]
 with localcontext() as ctx:
  ctx.prec=case['confirmationPrecisionDigits']
  def quantize(row,eligible):
   total=D(0);thresholds=[0]
   for j in range(n-1):
    total+=D(row[j]);prefix=[i for i in eligible if i<=j];suffix=[i for i in eligible if i>j]
    if not prefix:threshold=0
    elif not suffix:threshold=M
    elif sorted(energies[i] for i in prefix)==sorted(energies[i] for i in suffix):threshold=M//2
    else:threshold=int((total*M).to_integral_value(rounding=ROUND_CEILING))
    thresholds.append(threshold)
   thresholds.append(M)
   result=[b-a for a,b in zip(thresholds,thresholds[1:])];assert sum(result)==M and min(result)>=0
   return result
  K1=[quantize(row,[j for j,t in enumerate(states) if t[1]==states[i][1]]) for i,row in enumerate(r['kernels']['first'])]
  K2=[quantize(row,[j for j,t in enumerate(states) if t[0]==states[i][0]]) for i,row in enumerate(r['kernels']['second'])]
  QB=[quantize(r['target'],list(range(n))) for _ in states]
  if mode=='first':Q=K1;power=bits
  elif mode=='second':Q=K2;power=bits
  elif mode=='blocked':Q=QB;power=bits
  elif mode=='random':Q=[[a+b for a,b in zip(x,y)] for x,y in zip(K1,K2)];power=bits+1
  else:Q=[[sum(K1[i][k]*K2[k][j] for k in range(n)) for j in range(n)] for i in range(n)];power=2*bits
  bound=2 if mode=='systematic' else n-1 if mode=='blocked' else 1
  for exact,qrow in zip(r['kernels'][mode],Q):
   tv=sum(abs(D(a)/D(2**power)-D(b)) for a,b in zip(qrow,exact))/2
   assert tv<=D(bound)/M+D('1e-150'),(caseid,mode,tv)
  seed=initial_seed;draw_count=0
  def random_uint():
   global seed,draw_count
   seed=(1664525*seed+1013904223)%(2**32);draw_count+=1;return seed
  def draw(row):
   R=random_uint()>>(32-bits);total=0
   for i,m in enumerate(row):
    total+=m
    if R<total:return i
   raise AssertionError('row')
  current=0;path=[current];counts=[1]+[0]*(n-1)
  for t in range(steps):
   if mode=='first':current=draw(K1[current])
   elif mode=='second':current=draw(K2[current])
   elif mode=='blocked':current=draw(QB[current])
   elif mode=='random':current=draw(K1[current]) if random_uint()>>31==0 else draw(K2[current])
   else:current=draw(K2[draw(K1[current])])
   path.append(current);counts[current]+=1
  options=dict(seed=initial_seed,bits=bits,steps=steps,mode=mode)
  reference=dict(states=path,observedCounts=counts,finalPrngState=seed,randomDrawCount=draw_count,
   coordinatesResampled=steps*(2 if mode in ['systematic','blocked'] else 1),
   quantizedKernel=dict(denominatorPower=power,numerators=[[str(v) for v in row] for row in Q]),oneStepTVBound=dict(numerator=bound,denominatorPower=bits))
  rows.append(dict(id=f'{caseid}-{mode}-{bits}-{initial_seed}',parameters=p,options=options,reference=reference))
  print(json.dumps(dict(case=rows[-1]['id'],steps=steps)),flush=True)
Path('scripts/fixtures/finite-gibbs-path-oracle.json').write_text(json.dumps(dict(scope='Independent actual-input Decimal-CDF and integer LCG32-high-bits references; Q assumes iid grid draws, seeded trace is deterministic',cases=rows),indent=2)+'\n')
print(json.dumps(dict(cases=len(rows))))
