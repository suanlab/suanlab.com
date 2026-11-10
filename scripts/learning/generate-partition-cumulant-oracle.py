"""Independent Decimal reference for actual binary64 inputs; no TS imports.
The separate ideal-rational fixture is deliberately not used as this oracle.
"""
from decimal import Decimal as D, localcontext
from pathlib import Path
import json, math, sys
KB=1.380649e-23
cases=[]
def add(id,levels,T=1,offset=0):cases.append(dict(id=id,parameters=dict(energyLevels=levels,energyOffset=offset,temperature=T)))
gap=KB*math.log(2)
add('three-level-ln2',[0,gap,2*gap]);add('three-level-inversion',[0,gap,2*gap],-1)
add('offset-keeps-small-gaps',[0,1,2],1/KB,2**60)
add('already-rounded-flat-levels',[float(2**60+i) for i in range(3)],1/KB)
add('degenerate-zero',[0,0,0]);add('degenerate-shifted',[0,0,0],1,1e280)
add('degenerate-negative-T',[0,0,0],-3,-1e-23)
add('repeated-labeled-states',[0,0,gap,gap,gap]);add('repeated-inversion',[0,0,gap,gap,gap],-1)
add('asymmetric-four',[-2e-23,-1e-23,1e-23,3e-23],.75,1e-23)
add('asymmetric-four-negative',[-2e-23,-1e-23,1e-23,3e-23],-.75,1e-23)
add('single-subnormal',[0],float.fromhex('0x0.0000000000001p-1022'),float.fromhex('0x0.0000000000001p-1022'))
add('thirty-two-degenerate',[0]*32,float.fromhex('0x1.fffffffffffffp+1023'))
add('probability-underflows-energy-survives',[0,1e280],1e280/(KB*800))
add('hot-third-cumulant-rescue',[-1e70,0,1e70],float.fromhex('0x1.fffffffffffffp+1023'))
add('hot-six-state-cancellation',[-1e70,1e70,0,0,0,0],float.fromhex('0x1.fffffffffffffp+1023'))
small=float.fromhex('0x0.0000000000001p-1022')
add('subnormal-three-level',[0,small,2*small],float.fromhex('0x1.fffffffffffffp+1023'))
add('subnormal-six-state-cancellation',[-small,small,0,0,0,0],float.fromhex('0x1.fffffffffffffp+1023'))
add('zero-ground',[0,gap],0);add('zero-positive-ground',[0,gap],0,1)
add('zero-negative-ground',[0,gap],0,-1);add('zero-degenerate-ground',[0,0,gap],0)
add('zero-overflowed-ground',[float.fromhex('0x1.fffffffffffffp+1023')]*2,0,float.fromhex('0x1.fffffffffffffp+1023'))
add('signed-thirteen',[(i-6)*KB for i in range(13)],1.25,.3*KB)

def dec(x):return D.from_float(float(x))
def reference(p,precision):
 with localcontext() as c:
  c.prec=precision
  levels=[dec(x) for x in p['energyLevels']];offset=dec(p['energyOffset']);temp=dec(p['temperature']);kb=dec(KB)
  energies=[offset+x for x in levels]
  positive=temp>=0;anchor=min(energies) if positive else max(energies);g0=energies.count(anchor)
  if temp==0:
   probs=[D(1)/g0 if e==anchor else D(0) for e in energies];s=D(g0).ln();logz=s if anchor==0 else ('+infinity' if anchor<0 else '-infinity')
   return dict(meanEnergy=anchor,energyVariance=D(0),thirdEnergyCumulant=D(0),heatCapacity=D(0),heatCapacityInKB=D(0),entropyInKB=s,entropy=kb*s,freeEnergy=anchor,centeredLogPartition=s,logPartition=logz,probabilities=probs)
  tau=kb*abs(temp);sign=D(1) if positive else D(-1)
  costs=[sign*(e-anchor) for e in energies];weights=[(-g/tau).exp() for g in costs];total=sum(weights);probs=[w/total for w in weights]
  # Direct centered moments differ from the TS pair sum / polynomial enclosure.
  mean=anchor if all(e==anchor for e in energies) else sum(p*e for p,e in zip(probs,energies))
  var=sum(p*(e-mean)**2 for p,e in zip(probs,energies));third=sum(p*(e-mean)**3 for p,e in zip(probs,energies))
  logs={p:p.ln() for p in set(probs)}
  entropy=-sum(p*logs[p] for p in probs);lnw=total.ln();lnz=lnw-sign*anchor/tau
  return dict(meanEnergy=mean,energyVariance=var,thirdEnergyCumulant=third,heatCapacity=var/(kb*temp*temp),heatCapacityInKB=var/(tau*tau),entropyInKB=entropy,entropy=kb*entropy,freeEnergy=anchor-sign*tau*lnw,centeredLogPartition=lnw,logPartition=lnz,probabilities=probs)

def observation(q):
 if isinstance(q,str):return dict(kind='infinite',sign=1 if q[0]=='+' else -1)
 if q==0:return dict(kind='zero',value='0',sign=0,logMagnitude=None)
 with localcontext() as c:
  c.prec=100
  return dict(kind='finite',value=str(+q),sign=1 if q>0 else -1,logMagnitude=str(q.copy_abs().ln()))

def interval_endpoint(v):
 with localcontext() as c:
  c.prec=3300
  n=v['numerator'];d=v['denominator']
  return D(n['coefficient'])/D(d['coefficient'])*(D(2)**(n['exponent']-d['exponent']))

def precision_for(id):
 if id=='subnormal-six-state-cancellation':return 3200
 if id in ['hot-six-state-cancellation','subnormal-three-level']:return 1600
 if id in ['probability-underflows-energy-survives','hot-third-cumulant-rescue']:return 800
 return 400

rows=[]
for row in cases:
 precision=precision_for(row['id'])
 a=reference(row['parameters'],precision);b=reference(row['parameters'],precision+80)
 for key,q in a.items():
  seq=q if isinstance(q,list) else [q];other=b[key] if isinstance(b[key],list) else [b[key]]
  for v,w in zip(seq,other):
   if isinstance(v,str):assert v==w
   elif v==0:assert w==0,(row['id'],key,str(w))
   else:
    with localcontext() as c:
     c.prec=3300
     assert abs((v-w)/w)<D('1e-150'),(row['id'],key)
 print(json.dumps({'checked':row['id'],'precision':precision,'confirmed':precision+80}),file=sys.stderr,flush=True)
 rows.append(dict(**row,precisionDigits=precision,confirmationPrecisionDigits=precision+80,reference={k:[observation(x) for x in q] if isinstance(q,list) else observation(q) for k,q in b.items()}))
output={'scope':'Independent Decimal enumeration of actual binary64 common offsets, levels, temperatures and kB. Not ideal exp(log(2)) identities.','minimumPrecisionDigits':400,'maximumPrecisionDigits':3200,'confirmationExtraDigits':80,'cases':rows}
path=Path('scripts/fixtures/partition-cumulant-oracle.json')
if len(sys.argv)>2 and sys.argv[1]=='--check-model':
 actual=json.loads(Path(sys.argv[2]).read_text());checked=0
 for row in rows:
  model=actual[row['id']];truth=reference(row['parameters'],row['confirmationPrecisionDigits'])
  fields={k:v for k,v in truth.items() if k!='probabilities'}
  for key,q in fields.items():
   result=model[key]
   if result.get('bounds') and not isinstance(q,str):
    lo=interval_endpoint(result['bounds']['lower']);hi=interval_endpoint(result['bounds']['upper'])
    with localcontext() as c:
     c.prec=3300
     # Reference arithmetic is approximate. The per-case precision minus
     # 80 guard digits is still much tighter than a 48-bit observation.
     slack=max(q.copy_abs(),D('1e-10000'))*(D(10)**(-row['confirmationPrecisionDigits']+80))
     assert lo-slack<=q<=hi+slack,(row['id'],key,str(lo),str(q),str(hi))
    checked+=1
  for state,q in zip(model['states'],truth['probabilities']):
   if q>0 and state['logProbability'].get('bounds'):
    result=state['logProbability'];lo=interval_endpoint(result['bounds']['lower']);hi=interval_endpoint(result['bounds']['upper'])
    with localcontext() as c:
     c.prec=row['confirmationPrecisionDigits'];lp=q.ln()
     slack=max(lp.copy_abs(),D('1e-10000'))*(D(10)**(-row['precisionDigits']+80))
     assert lo-slack<=lp<=hi+slack,(row['id'],'logProbability')
    checked+=1
   result=state['probability']
   if result.get('bounds'):
    lo=interval_endpoint(result['bounds']['lower']);hi=interval_endpoint(result['bounds']['upper'])
    with localcontext() as c:
     c.prec=3300;slack=max(q.copy_abs(),D('1e-10000'))*(D(10)**(-row['confirmationPrecisionDigits']+80))
     assert lo-slack<=q<=hi+slack,(row['id'],'probability')
    checked+=1
 print(json.dumps({'cases':len(rows),'enclosuresChecked':checked,'maximumDecimalPrecision':3280,'minimumDecimalPrecision':480}))
else:
 path.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'cases':len(rows),'minimumPrecision':400,'maximumPrecision':3200,'confirmationExtra':80,'fixture':str(path)}))
