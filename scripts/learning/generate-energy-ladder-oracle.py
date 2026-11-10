"""Independent Decimal finite enumeration and hyperbolic infinite reference.
Actual binary64 inputs and stored kB; ideal log-two fractions are not an oracle.
"""
from decimal import Decimal as D, localcontext
from pathlib import Path
import json, math
KB=1.380649e-23
cases=[]
def add(id,gap,T,L,precision=400):cases.append(dict(id=id,parameters=dict(gapEnergy=gap,temperature=T,cutoff=L),precisionDigits=precision))
add('near-log-two',KB*math.log(2),1,2)
add('one-state-cutoff',KB*math.log(2),1,0)
add('finite-negative-temperature',KB*.7,-1,5)
add('zero-gap',0,2,9)
add('zero-gap-negative-T',0,-2,9)
add('zero-temperature',KB,0,8)
add('zero-gap-zero-temperature',0,0,4)
add('single-zero-energy-state',0,1,0)
add('hot-q-rounds-to-one',KB*1e-30,1,31)
add('subnormal-gap-hot',float.fromhex('0x0.0000000000001p-1022'),float.fromhex('0x1.fffffffffffffp+1023'),2,1600)
add('underflow-probability-rescued-energy',1e280,1e280/(KB*800),0,800)
add('large-cutoff',KB*.7,1,4096)
add('moderate-asymmetric-cutoff',KB*.17,.8,16)
add('negative-T-large-cutoff',KB*1.2,-1,300)
add('cold-leading-moments',KB*5000,1,2,2600)
add('zero-subnormal-ground',float.fromhex('0x0.0000000000001p-1022'),0,2)

def dec(v):return D.from_float(float(v))
def sample_indices(L):
 if L<=62:return list(range(L+2))
 # JS Math.round uses ties towards +infinity; indices here are nonnegative.
 return sorted(set(list(range(16))+[L-n for n in range(16)]+[math.floor(n*L/30+.5) for n in range(31)]+[L+1]))
def reference(p,precision):
 with localcontext() as ctx:
  ctx.prec=precision
  epsilon,T=dec(p['gapEnergy']),dec(p['temperature']);L=p['cutoff'];m=L+1;kb=dec(KB)
  ns=list(map(D,range(m)));energies=[epsilon*(n+D('.5')) for n in ns]
  if epsilon==0 or T==0:weights=[D(1) if epsilon==0 or n==0 else D(0) for n in ns]
  else:
   anchor=min(energies) if T>0 else max(energies)
   weights=[(-(e-anchor)/(kb*T)).exp() for e in energies]
  W=sum(weights);probs=[w/W for w in weights]
  mean=sum(p*n for p,n in zip(probs,ns));var=sum(p*(n-mean)**2 for p,n in zip(probs,ns))
  entropy=-sum(p*p.ln() for p in probs if p)
  U=epsilon*(mean+D('.5'));variance=epsilon**2*var
  lnz='-infinity' if epsilon>0 and T==0 else W.ln() if epsilon==0 else W.ln()-anchor/(kb*T)
  F=U if T==0 else -kb*T*lnz
  C=D(0) if T==0 else variance/(kb*T)**2
  finite=dict(meanIndex=mean,indexVariance=var,meanEnergy=U,energyVariance=variance,heatCapacityInKB=C,heatCapacity=kb*C,entropyInKB=entropy,entropy=kb*entropy,freeEnergy=F,logPartition=lnz)
  infinite=None;comparison=None
  if epsilon>0 and T>0:
   x=epsilon/(kb*T);t=x/2;ep=t.exp();em=(-t).exp();twosinh=ep-em;coth=(ep+em)/twosinh
   meaninf=(coth-1)/2;varinf=1/twosinh**2;Uinf=epsilon*coth/2;Cinf=(x/twosinh)**2
   logZinf=-twosinh.ln();Sinf=x*coth/2+logZinf
   infinite=dict(meanIndex=meaninf,indexVariance=varinf,meanEnergy=Uinf,energyVariance=epsilon**2*varinf,heatCapacityInKB=Cinf,heatCapacity=kb*Cinf,entropyInKB=Sinf,entropy=kb*Sinf,freeEnergy=-kb*T*logZinf,logPartition=logZinf)
   delta=(-m*x).exp();retained=1-delta;bias=m*delta/retained;vBias=m*m*delta/(retained**2)
   # ln(1-delta) must survive when 1-delta rounds to one in Decimal too.
   if delta<D('.1'):
    logretained=-delta;power=delta;degree=1
    while True:
     degree+=1;power*=delta;term=power/degree;logretained-=term
     if term.copy_abs()<delta*(D(10)**(-precision-10)):break
   else:logretained=retained.ln()
   comparison=dict(missingMass=delta,retainedMass=retained,logRetainedMass=logretained,totalVariation=delta,forwardKL=-logretained,reverseKL='+infinity',meanIndexBias=bias,indexVarianceBias=vBias,meanEnergyBias=epsilon*bias)
   infprobs=[(1-(-x).exp())*(-n*x).exp() for n in sample_indices(L)]
  elif epsilon>0 and T==0:
   infinite=dict(finite,meanIndex=D(0),indexVariance=D(0),entropyInKB=D(0),entropy=D(0))
   comparison={k:D(0) for k in ['missingMass','logRetainedMass','totalVariation','forwardKL','reverseKL','meanIndexBias','indexVarianceBias','meanEnergyBias']};comparison['retainedMass']=D(1)
   infprobs=[D(int(n==0)) for n in sample_indices(L)]
  else:infprobs=[None]*len(sample_indices(L))
  samples=[dict(n=n,finiteProbability=probs[n] if n<=L else D(0),infiniteProbability=pi) for n,pi in zip(sample_indices(L),infprobs)]
  return dict(finite=finite,infinite=infinite,comparison=comparison,samples=samples)

def flatten(v,path=()):
 if isinstance(v,dict):
  for k,x in v.items():yield from flatten(x,path+(k,))
 elif isinstance(v,list):
  for i,x in enumerate(v):yield from flatten(x,path+(i,))
 else:yield path,v
rows=[]
for case in cases:
 precision=case['precisionDigits'];a=dict(flatten(reference(case['parameters'],precision)));truth=reference(case['parameters'],precision+80)
 with localcontext() as c:
  c.prec=precision+80
  for path,y in flatten(truth):
   x=a[path]
   if isinstance(y,D):
    if y==0:assert x==0,(case['id'],path)
    else:assert abs((x-y)/y)<D('1e-100'),(case['id'],path,str(x),str(y))
   else:assert x==y
 def stringify(v):
  if isinstance(v,dict):return {k:stringify(x) for k,x in v.items()}
  if isinstance(v,list):return [stringify(x) for x in v]
  return str(v) if isinstance(v,D) else v
 rows.append(dict(**case,confirmationPrecisionDigits=precision+80,reference=stringify(truth)))
 print(case['id'],precision,precision+80,flush=True)
Path('scripts/fixtures/energy-ladder-oracle.json').write_text(json.dumps(dict(scope='Independent Decimal direct finite enumeration; infinite hyperbolic partition and centered thermodynamics; actual binary64 inputs',cases=rows),indent=2)+'\n')
