"""Independent Decimal enumeration; no TypeScript or core formula imports."""
from decimal import Decimal as D, localcontext
from pathlib import Path
import json, math
KB=1.380649e-23
cases=[]
def add(id,J,h,T=1,support='full-product',precision=400):
 cases.append(dict(id=id,parameters=dict(couplingEnergy=J,fieldEnergy=h,temperature=T,support=support),precisionDigits=precision))
add('ideal-near-ln2',KB*math.log(2),0)
add('antiferromagnetic',-KB*math.log(2),0)
add('biased',.8*KB,.35*KB)
add('negative-temperature',.8*KB,.35*KB,-1)
add('independent-biased',0,.35*KB)
add('independent-uniform',0,0)
add('constrained-independent',0,0,support='fixed-zero-sum')
add('constrained-biased',2*KB,100*KB,support='fixed-zero-sum')
add('constrained-negative-temperature',-2*KB,3*KB,-2,'fixed-zero-sum')
add('weak-coupling',1e-60,0)
add('much-weaker-coupling',1e-200,0,precision=600)
add('zero-ferro-ground',1,0,0)
add('zero-anti-ground',-1,0,0)
add('zero-three-way-crossing',-1,1,0)
add('zero-unique-ground',1,.3,0)
add('zero-restricted',2,10,0,'fixed-zero-sum')

def dec(v):return D.from_float(float(v))
def reference(p,precision):
 with localcontext() as c:
  c.prec=precision
  J,h,T=map(dec,[p['couplingEnergy'],p['fieldEnergy'],p['temperature']]);kb=dec(KB)
  spins=[(-1,-1),(-1,1),(1,-1),(1,1)]
  if p['support']=='fixed-zero-sum':spins=[s for s in spins if sum(s)==0]
  feat=[[D(a*b),D(a+b),D(a),D(b)] for a,b in spins]
  energies=[-J*f[0]-h*f[1] for f in feat]
  anchor=min(energies) if T>=0 else max(energies)
  if T==0:weights=[D(int(e==anchor)) for e in energies]
  else:weights=[(-(e-anchor)/(kb*T)).exp() for e in energies]
  total=sum(weights);probs=[w/total for w in weights]
  means=[sum(p*f[i] for p,f in zip(probs,feat)) for i in range(4)]
  cov=[[sum(p*(f[i]-means[i])*(f[j]-means[j]) for p,f in zip(probs,feat)) for j in range(4)] for i in range(4)]
  marginals=[[sum(p for p,s in zip(probs,spins) if s[i]==sign) for sign in [-1,1]] for i in range(2)]
  mi=sum(p*(p/(marginals[0][int(s[0]==1)]*marginals[1][int(s[1]==1)])).ln() for p,s in zip(probs,spins) if p)
  # Symmetry/independence proofs avoid assigning meaning to Decimal roundoff.
  if h==0:means[1:]=[D(0),D(0),D(0)];cov[0][1]=cov[1][0]=D(0)
  if J==0 and p['support']=='full-product':mi=D(0);cov[2][3]=cov[3][2]=D(0)
  out=dict(meanAlignment=means[0],meanMagnetization=means[1],spinCovariance=cov[2][3],mutualInformation=mi,sourceHessian=[row[:2] for row in cov[:2]],probabilities=probs)
  if T!=0:
   lnz=total.ln()-anchor/(kb*T)
   independent_anchor=-abs(h) if T>0 else abs(h)
   independent_weights=[(-(e-independent_anchor)/(kb*T)).exp() for e in [-h,h]]
   lnz1=sum(independent_weights).ln()-independent_anchor/(kb*T)
   defect=lnz-2*lnz1
   if J==0 and p['support']=='full-product':defect=D(0)
   out.update(logFactorizationDefect=defect,freeEnergyDefect=-kb*T*defect)
  return out

def leaves(x):
 if isinstance(x,list):
  for v in x:yield from leaves(v)
 else:yield x
rows=[]
for case in cases:
 precision=case['precisionDigits'];a=reference(case['parameters'],precision);truth=reference(case['parameters'],precision+80)
 with localcontext() as c:
  c.prec=precision+80
  for key in a:
   for x,y in zip(leaves(a[key]),leaves(truth[key])):
    if y==0:assert abs(x)<D(10)**(-precision+20),(case['id'],key)
    else:assert abs((x-y)/y)<D('1e-100'),(case['id'],key)
 def strings(x):return [strings(v) for v in x] if isinstance(x,list) else str(x)
 rows.append(dict(**case,confirmationPrecisionDigits=precision+80,reference={k:strings(v) for k,v in truth.items()}))
 print(case['id'],precision,precision+80,flush=True)
Path('scripts/fixtures/coupled-partition-oracle.json').write_text(json.dumps(dict(scope='Independent high precision enumeration of actual binary64 Hamiltonian inputs',cases=rows),indent=2)+'\n')
