"""Independent references for the next weighted-Boltzmann family.
Positive real base weights are not automatically physical degeneracies.
All binary64 inputs and the stored kB are interpreted. Model validation and
reader integration are recorded separately from reference generation.
"""
from decimal import Decimal as D, localcontext
from pathlib import Path
import json,math
KB=1.380649e-23;gap=KB*math.log(2);small=float.fromhex('0x0.0000000000001p-1022');large=float.fromhex('0x1.fffffffffffffp+1023')
cases=[]
def add(id,levels,weights,T=1,offset=0,precision=400):
 cases.append(dict(id=id,parameters=dict(energyLevels=levels,referenceWeights=weights,energyOffset=offset,temperature=T),precisionDigits=precision))
add('counting-multiplicity',[0,gap,2*gap],[1,2,1])
add('normalized-prior',[0,gap,2*gap],[.25,.5,.25])
add('large-reference-scale',[0,gap,2*gap],[2.**900,2.**901,2.**900])
add('small-reference-scale',[0,gap,2*gap],[2.**-900,2.**-899,2.**-900])
add('noninteger-reference',[0,gap,2*gap],[.1,.2,.7])
add('forbidden-middle',[0,gap,2*gap],[1,0,1])
add('forbidden-low-energy',[0,-large,gap],[1,0,1])
add('single-allowed-label',[0,gap,2*gap],[0,3,0])
add('rare-mass-physical-energy',[0,1e280],[1,2],1e280/(KB*800),precision=800)
add('common-offset-keeps-gaps',[0,gap,2*gap],[1,2,1],offset=2**60)
add('extreme-reference-ratio',[0,gap,2*gap],[large,small,1],precision=1600)
add('subnormal-gap-hot',[0,small,2*small],[1,2,1],large,precision=1600)
add('bounded-negative-temperature',[0,gap,2*gap],[1,2,1],-1)
add('zero-weighted-ground',[0,0,gap],[1,3,.25],0)
add('zero-prior-ground',[0,0,gap],[.25,.5,.25],0)
add('all-levels-equal',[0,0,0],[.25,.5,.25],1)

def dec(x):return D.from_float(float(x))
def reference(parameters,precision):
 with localcontext() as c:
  c.prec=precision
  levels=list(map(dec,parameters['energyLevels']));weights=list(map(dec,parameters['referenceWeights']));offset=dec(parameters['energyOffset']);T=dec(parameters['temperature']);kb=dec(KB)
  assert len(levels)==len(weights)>0 and all(w>=0 for w in weights) and sum(weights)>0
  energies=[offset+e for e in levels];allowed=[i for i,m in enumerate(weights) if m>0]
  anchor=min(energies[i] for i in allowed) if T>=0 else max(energies[i] for i in allowed)
  if T==0:raw=[m if energies[i]==anchor else D(0) for i,m in enumerate(weights)]
  else:raw=[m*(-(e-anchor)/(kb*T)).exp() if m else D(0) for m,e in zip(weights,energies)]
  W=sum(raw);probs=[w/W for w in raw]
  mean=anchor if all(energies[i]==anchor for i in allowed) else sum(p*e for p,e in zip(probs,energies))
  variance=sum(p*(e-mean)**2 for p,e in zip(probs,energies))
  labelEntropy=-sum(p*p.ln() for p in probs if p)
  correction=sum(p*m.ln() for p,m in zip(probs,weights) if p)
  relativeEntropy=labelEntropy+correction
  if T==0:
   capacity=D(0);freeEnergy=anchor;logZ=W.ln() if anchor==0 else ('+infinity' if anchor<0 else '-infinity')
  else:capacity=variance/(kb*T)**2;logZ=W.ln()-anchor/(kb*T);freeEnergy=anchor-kb*T*W.ln()
  return dict(referenceMeasureTotal=sum(weights),meanEnergy=mean,energyVariance=variance,labelEntropyNats=labelEntropy,meanLogReferenceWeight=correction,relativeEntropyInKB=relativeEntropy,heatCapacityInKB=capacity,freeEnergy=freeEnergy,logPartition=logZ,probabilities=probs,logProbabilities=[p.ln() if p else '-infinity' for p in probs])

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
    if y==0:assert abs(x)<D(10)**(-precision+30),(case['id'],path)
    else:assert abs((x-y)/y)<D('1e-150'),(case['id'],path,str(x),str(y))
   else:assert x==y
 def stringify(v):
  if isinstance(v,dict):return {k:stringify(x) for k,x in v.items()}
  if isinstance(v,list):return [stringify(x) for x in v]
  return str(v) if isinstance(v,D) else v
 rows.append(dict(**case,confirmationPrecisionDigits=precision+80,reference=stringify(truth)))
 print(json.dumps(dict(case=case['id'],precision=precision,confirmation=precision+80)),flush=True)
Path('scripts/fixtures/weighted-boltzmann-oracle.json').write_text(json.dumps(dict(scope='Actual binary64 independent Decimal references; model and reader validation evidence is recorded separately',cases=rows),indent=2)+'\n')
print(json.dumps(dict(cases=len(rows),referenceOnly=True)))
