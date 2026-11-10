"""Independent Decimal Landau references from analytic curvature turning points.
No Sturm counts, rational gcd chains or TypeScript interval code is imported.
"""
from decimal import Decimal as D,localcontext
from math import nextafter,inf
from pathlib import Path
import json
cases=[]
def add(id,a,b,c=1,H=0,precision=400):cases.append(dict(id=id,parameters=dict(quadraticA=a,quarticB=b,sexticC=c,fieldH=H),precisionDigits=precision))
add('single-well',1,1)
add('ordered-continuous',-1,1)
add('quartic-critical',0,1)
add('tricritical-exact',0,0)
add('tricritical-ordered',-1,0)
add('quartic-flat-maximum',0,-1)
add('three-phase-coexistence',3/16,-1)
add('below-coexistence',nextafter(3/16,-inf),-1)
add('above-coexistence',nextafter(3/16,inf),-1)
add('ordered-spinodal',1/4,-1)
add('below-ordered-spinodal',nextafter(1/4,-inf),-1)
add('above-ordered-spinodal',nextafter(1/4,inf),-1)
add('tilted-ordered',-1,1,H=.1)
add('tilted-reflected',-1,1,H=-.1)
add('three-well-tilted',.2,-1,H=.005)
add('exact-tilted-fold',-5/16,0,H=-1/8)
add('below-tilted-fold',-5/16,0,H=nextafter(-1/8,-inf))
add('above-tilted-fold',-5/16,0,H=nextafter(-1/8,inf))
add('exact-tilted-quartic-minimum',15/16,-2.5,3,.25)
add('outside-spin-domain',-4,-4,.25,4)
add('maximum-coefficients',4,4,4,4)
tiny=float.fromhex('0x0.0000000000001p-1022')
add('tricritical-subnormal-field',0,0,H=tiny,precision=1200)
add('coexistence-subnormal-field',3/16,-1,H=tiny,precision=1200)
add('subnormal-quadratic',tiny,-1,precision=1200)
add('subnormal-quartic-critical',0,tiny,precision=1200)

def reference(p,digits):
 with localcontext() as ctx:
  ctx.prec=digits
  a,b,c,H=[D.from_float(float(p[k])) for k in ['quadraticA','quarticB','sexticC','fieldH']]
  tolerance=D('1e-500') if digits>=1200 else D('1e-220')
  def g(m):return c*m**5+b*m**3+a*m-H
  def phi(m):return a*m*m/2+b*m**4/4+c*m**6/6-H*m
  def deriv(m,k):
   coefficients=[-H,a,D(0),b,D(0),c]
   for _ in range(k):coefficients=[i*v for i,v in enumerate(coefficients)][1:]
   result=D(0)
   for v in reversed(coefficients):result=result*m+v
   return result
  def bisect(lo,hi):
   flo=g(lo);fhi=g(hi);assert flo*fhi<0
   for _ in range(6000):
    mid=(lo+hi)/2;fm=g(mid)
    if not fm:return mid
    if fm*flo>0:lo=mid;flo=fm
    else:hi=mid
    if (lo>0 or hi<0) and hi-lo<=min(abs(lo),abs(hi))*tolerance:return (lo+hi)/2
   raise AssertionError('Reference bisection exhausted')
  special={}
  if not H:
   ms=[D(0)];special[D(0)]=5 if not a and not b else 3 if not a else 1
   disc=b*b-4*a*c
   if disc>=0:
    for y in set([(-b-disc.sqrt())/(2*c),(-b+disc.sqrt())/(2*c)]):
     if y>0:
      m=y.sqrt();ms.extend([-m,m]);special[-m]=special[m]=2 if not disc else 1
   ms.sort()
  else:
   turning=[];disc=9*b*b-20*a*c
   if disc>=0:
    for y in set([(-3*b-disc.sqrt())/(10*c),(-3*b+disc.sqrt())/(10*c)]):
     if y>0:turning.extend([-y.sqrt(),y.sqrt()])
   R=D(2)+max(abs(a/c),abs(b/c),abs(H/c));ends=[-R]+sorted(turning)+[R]
   ms=[]
   for m in ends:
    if not g(m):
     ms.append(m)
     k=1
     while not deriv(m,k):k+=1
     special[m]=k
   for lo,hi in zip(ends,ends[1:]):
    if g(lo)*g(hi)<0:ms.append(bisect(lo,hi))
   ms.sort()
  coexist=not H and b<0 and 16*a*c==3*b*b
  rows=[]
  for m in ms:
   mult=special.get(m,1);leading=deriv(m,mult)
   local='stationary-inflection' if mult%2==0 else 'minimum' if leading>0 else 'maximum'
   energy=D(0) if not m or coexist and local=='minimum' else phi(m)
   curve=D(0) if mult>1 else deriv(m,1)
   response=None if mult%2==0 else 'positive-infinity' if mult>1 and leading>0 else '-infinity' if mult>1 else 1/curve
   rows.append(dict(magnetization=m,curvature=curve,constrainedFreeEnergy=energy,leadingStationaryDerivative=leading,
    stationaryMultiplicity=mult,localKind=local,branchFieldDerivative=response))
  minimum=min(x['constrainedFreeEnergy'] for x in rows if x['localKind']=='minimum')
  indices=[i for i,x in enumerate(rows) if x['localKind']=='minimum' and x['constrainedFreeEnergy']==minimum]
  # At zero field symmetric pairs are analytic equalities, not rounded ties.
  if not H:
   for i,x in enumerate(rows):
    if x['localKind']=='minimum' and any(ms[j]==-ms[i] for j in indices) and i not in indices:indices.append(i)
   indices.sort()
  for i,x in enumerate(rows):
   x['globalMinimum']=i in indices
   x['role']='global-minimum' if i in indices else 'metastable-minimum' if x['localKind']=='minimum' else 'unstable-maximum' if x['localKind']=='maximum' else 'stationary-inflection'
   x['freeEnergyExcess']=D(0) if i in indices else x['constrainedFreeEnergy']-minimum
  barriers=[]
  for j,x in enumerate(rows):
   if x['localKind']=='maximum':
    for direction in [-1,1]:
     i=j+direction
     while 0<=i<len(rows) and rows[i]['localKind']=='stationary-inflection':i+=direction
     if 0<=i<len(rows) and rows[i]['localKind']=='minimum':barriers.append(dict(minimumIndex=i,maximumIndex=j,barrier=x['constrainedFreeEnergy']-rows[i]['constrainedFreeEnergy']))
  return dict(rootCount=len(rows),realRootMultiplicity=sum(x['stationaryMultiplicity'] for x in rows),roots=rows,globalMinimumIndices=indices,minimumFreeEnergy=minimum,barriers=barriers,exactCoexistence=bool(coexist))

def flatten(x,path=()):
 if isinstance(x,dict):
  for k,v in x.items():yield from flatten(v,path+(k,))
 elif isinstance(x,list):
  for i,v in enumerate(x):yield from flatten(v,path+(i,))
 else:yield path,x
def stringify(x):
 if isinstance(x,D):return str(x)
 if isinstance(x,dict):return {k:stringify(v) for k,v in x.items()}
 if isinstance(x,list):return [stringify(v) for v in x]
 return x
rows=[]
for case in cases:
 digits=case['precisionDigits'];low=dict(flatten(reference(case['parameters'],digits)));truth=reference(case['parameters'],digits+80)
 with localcontext() as ctx:
  ctx.prec=digits+80
  for path,y in flatten(truth):
   x=low[path]
   if isinstance(y,D):assert abs(x-y)<=abs(y)*D('1e-150')+D(10)**(-digits+60),(case['id'],path,str(x),str(y))
   else:assert x==y,(case['id'],path,x,y)
 rows.append(dict(**case,confirmationPrecisionDigits=digits+80,reference=stringify(truth)))
 print(json.dumps(dict(case=case['id'],confirmedDigits=digits+80)),flush=True)
Path('scripts/fixtures/landau-sixth-order-oracle.json').write_text(json.dumps(dict(scope='Independent analytic turning-point Decimal reference only; no reader/model completion claim',cases=rows),indent=2)+'\n')
print(json.dumps(dict(referenceCases=len(rows))))
