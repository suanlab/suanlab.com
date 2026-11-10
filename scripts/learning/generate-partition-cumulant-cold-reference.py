"""Independent analytic cold-tail logarithms and universal omission bounds.
These are leading-log references with a proved omission bound, not exact
transcendental values or a certification of floating logarithm evaluations.
"""
from fractions import Fraction as F
from decimal import Decimal as D, localcontext
from pathlib import Path
import json, math
KB=1.380649e-23;small=float.fromhex('0x0.0000000000001p-1022');large=float.fromhex('0x1.fffffffffffffp+1023')
cases=[('first-log-overflow',[0,large],small,0),('ground-degeneracy',[0,0,large],small,0),('cold-negative',[0,81930],-10/KB,0),('tiny-log-difference',[-81930,0,small],10/KB,0),('compensated-reference',[-81930,0,small],10/KB,81930),('just-beyond-threshold',[0,KB*8193],1,0)]
R=F(32,1<<8192);G=F(1<<2100);tolerance=F(1,1<<100)
assert R*(1+G*G)<tolerance
assert R+3*R*G+2*R*R*G**3<tolerance
assert R<tolerance

def dec(q):return D(q.numerator)/D(q.denominator)
def ref(levels,T,offset,precision):
 with localcontext() as ctx:
  ctx.prec=precision
  t=F.from_float(float(T));tau=F.from_float(KB)*abs(t);e=[F.from_float(float(offset))+F.from_float(float(x)) for x in levels]
  direction=1 if t>0 else -1;anchor=min(e) if direction>0 else max(e);g=[direction*(x-anchor) for x in e];g0=g.count(F(0));exc=[x for x in g if x]
  u=[dec(x/tau) for x in g];assert all(x==0 or x>D(8192) for x in u)
  def lse(values):
   maximum=max(values);return maximum+sum((x-maximum).exp() if x-maximum>=-8192 else D(0) for x in values).ln()
  logtau=dec(tau).ln();lng0=D(g0).ln();logr=lse([-x for x in u if x])
  def moment(power):return lse([-dec(x/tau)+D(power)*dec(x).ln() for x in exc])-lng0
  logs={'energyVariance':moment(2),'thirdEnergyCumulant':moment(3),'heatCapacityInKB':moment(2)-2*logtau,'heatCapacity':moment(2)-2*logtau+D.from_float(KB).ln()}
  signs={key:direction if key=='thirdEnergyCumulant' else 1 for key in logs}
  if anchor==0:logs['meanEnergy']=moment(1);signs['meanEnergy']=direction
  if g0==1:
   logs['entropyInKB']=lse([-x+(1+x).ln() for x in u if x]);signs['entropyInKB']=1
   logs['entropy']=logs['entropyInKB']+D.from_float(KB).ln();signs['entropy']=1
   logs['centeredLogPartition']=logr;signs['centeredLogPartition']=1
   if anchor==0:logs['logPartition']=logr;signs['logPartition']=1;logs['freeEnergy']=logr+logtau;signs['freeEnergy']=-direction
  state_logs=[]
  for x in u:
   if x==0 and g0==1:state_logs.append({'value':None,'logMagnitude':str(logr),'sign':-1})
   else:
    lp=-x-lng0;state_logs.append({'value':str(lp),'logMagnitude':str((-lp).ln()),'sign':-1})
  return {'logs':{key:{'logMagnitude':str(value),'sign':signs[key]} for key,value in logs.items()},'logProbabilities':state_logs,'anchorMultiplicity':g0}
rows=[]
for id,levels,T,offset in cases:
 a=ref(levels,T,offset,2400);b=ref(levels,T,offset,2480)
 for key,v in a['logs'].items():
  with localcontext() as c:
   c.prec=2500;x=D(v['logMagnitude']);y=D(b['logs'][key]['logMagnitude']);assert abs(x-y)<=max(D(1),abs(y))*D('1e-150')
 with localcontext() as c:
  c.prec=100
  compact={'logs':{k:{'logMagnitude':str(+D(v['logMagnitude'])),'sign':v['sign']} for k,v in b['logs'].items()},'logProbabilities':[{k:(str(+D(v)) if k!='sign' and v is not None else v) for k,v in p.items()} for p in b['logProbabilities']],'anchorMultiplicity':b['anchorMultiplicity']}
 rows.append({'id':id,'parameters':{'energyLevels':levels,'temperature':T,'energyOffset':offset},'reference':compact})
output={'contract':'Analytic cold leading-moment logarithms for actual binary64 inputs. Universal omitted relative physical corrections <2^-100. Decimal logarithms and normalized log-sums are approximations, with 80 extra precision digits checked. Not a floating transcendental certificate.','precisionDigits':2400,'confirmationPrecisionDigits':2480,'universalOmissionBoundsChecked':True,'cases':rows}
Path('scripts/fixtures/partition-cumulant-cold-reference.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({'cases':len(rows),'universalOmissionBoundsChecked':True,'precisionDigits':2400,'confirmed':2480}))
