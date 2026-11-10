"""Independent Fraction matrix powers, path sums and Cesaro formulas."""
from decimal import Decimal, localcontext
from fractions import Fraction as F
from itertools import product
from math import gcd
from pathlib import Path
import json

CASES = [
 ('asymmetric-from-zero',1,2,4,0,1,12),('asymmetric-from-one',1,2,4,1,1,12),
 ('symmetric-reset',1,1,2,0,1,8),('asymmetric-reset',1,3,4,2,3,12),
 ('alternating-point',1,1,1,0,1,12),('alternating-third',1,1,1,1,3,11),
 ('alternating-stationary',1,1,1,1,2,512),('identity-third',0,0,1,1,3,12),
 ('identity-maximum-denominator',0,0,1000000,1,1000000,512),
 ('absorbing-zero',0,1,2,1,1,12),('absorbing-one',1,0,2,0,1,12),
 ('immediate-absorbing-zero',0,1,1,1,1,8),('immediate-absorbing-one',1,0,1,0,1,8),
 ('absorbing-already-stationary',0,1,2,0,1,0),
 ('slow-switching',1,1,1000000,0,1,512),('near-alternation',999999,999999,1000000,0,1,512),
 ('tiny-positive-eigenvalue',500000,499999,1000000,0,1,512),
 ('tiny-negative-eigenvalue',500000,500001,1000000,0,1,512),
 ('scaled-transition',3,6,12,0,1,12),('scaled-initial',1,2,4,2,6,12),
 ('unbalanced-reset',1,999999,1000000,1,1,12),
 ('stationary-start-zero-records',1,1,3,1,2,0),('stationary-start',1,2,4,1,3,64),
 ('one-no-hold',1000000,1,1000000,0,1,64),('other-no-hold',1,1000000,1000000,1,1,64),
 ('zero-record-range',1,2,4,2,3,0),('quarter-equality',1,1,4,0,1,8),
 ('absorbing-quarter-not-recorded',0,1,2,1,1,1),
 ('alternating-maximum-integers',1000000,1000000,1000000,1,3,511),
 ('initial-maximum-denominator',1,2,4,999999,1000000,64),
]


def multiply(a,b):
 return [[sum(a[i][k]*b[k][j] for k in range(2)) for j in range(2)] for i in range(2)]


def advance(mu,p):
 return [sum(mu[i]*p[i][j] for i in range(2)) for j in range(2)]


def structure(p):
 reachable=[]
 for start in range(2):
  seen,stack={start},[start]
  while stack:
   i=stack.pop()
   for j in range(2):
    if p[i][j] and j not in seen:seen.add(j);stack.append(j)
  reachable.append(seen)
 classes=sorted({tuple(j for j in range(2) if j in reachable[i] and i in reachable[j]) for i in range(2)})
 closed=[list(g) for g in classes if all(p[i][j]==0 for i in g for j in range(2) if j not in g)]
 periods=[0,0];power=[[F(i==j) for j in range(2)] for i in range(2)]
 for t in range(1,7):
  power=multiply(power,p)
  for i in range(2):
   if power[i][i]>0:periods[i]=gcd(periods[i],t)
 return dict(irreducible=len(classes)==1,communicatingClasses=[list(g) for g in classes],closedCommunicatingClasses=closed,
             returnPeriods=[v if v else None for v in periods],uniqueClosedClassPeriod=periods[closed[0][0]] if len(closed)==1 else None)


def quantity(q):
 with localcontext() as ctx:
  ctx.prec=100
  log=str((Decimal(abs(q.numerator))/Decimal(q.denominator)).ln()) if q else None
 value=float(q)
 return dict(numerator=str(q.numerator),denominator=str(q.denominator),value=None if q and value==0 else value,
             sign=1 if q>0 else -1 if q<0 else 0,logMagnitude=log)


def serialize(v):
 if isinstance(v,F):return quantity(v)
 if isinstance(v,list):return [serialize(x) for x in v]
 if isinstance(v,dict):return {k:serialize(x) for k,x in v.items()}
 return v


records=[];time_checks=0;path_checks=0
for name,A,B,D,U,V,N in CASES:
 a,b,u=F(A,D),F(B,D),F(U,V);p=[[1-a,a],[b,1-b]];initial=[1-u,u]
 stationary=[b/(a+b),a/(a+b)] if a+b else None
 graph=structure(p);unique=len(graph['closedCommunicatingClasses'])==1
 assert unique==(stationary is not None)
 lam=1-a-b;initial_stationary=advance(initial,p)==initial
 periodic=graph['uniqueClosedClassPeriod']==2
 has_limit=not periodic or initial_stationary
 limit=stationary if unique and has_limit else initial if not unique else None
 classification=dict(**graph,stationaryIsUnique=unique,allStationaryLawsSatisfyDetailedBalance=True,
                     initialDistributionIsStationary=initial_stationary,hasPointwiseLimitForThisInitial=has_limit,
                     convergesToUniqueStationaryFromEveryInitial=unique and not periodic,
                     convergesToUniqueStationaryForThisInitial=has_limit if unique else None)
 power=[[F(i==j) for j in range(2)] for i in range(2)];sum_one=F(0);geometric=F(0);first_quarter=None;previous_tv=None
 history=[];selected=set(range(5))|{7,8,11,12,15,31,63,127,255,511,N}
 for t in range(N+1):
  mu=advance(initial,power);closed=u+(a-(a+b)*u)*geometric
  assert mu[1]==closed
  if stationary:assert closed==stationary[1]+(u-stationary[1])*lam**t
  sum_one+=mu[1];average=sum_one/(t+1)
  closed_average=stationary[1]+(u-stationary[1])*(1-lam**(t+1))/((t+1)*(1-lam)) if stationary else u
  assert average==closed_average
  tv=abs(mu[1]-stationary[1]) if stationary else None
  worst=max(abs(row[1]-stationary[1]) for row in power) if stationary else None
  if worst is not None and first_quarter is None and worst<=F(1,4):first_quarter=t
  if tv is not None:
   assert tv==abs(u-stationary[1])*abs(lam)**t and worst==max(stationary)*abs(lam)**t
   if previous_tv is not None:assert tv<=previous_tv
   previous_tv=tv
  assert sum(mu)==1 and min(mu)>=0
  if t<=5 and D<=12:
   enumeration=[F(0),F(0)]
   for path in product(range(2),repeat=t+1):
    probability=initial[path[0]]
    for i,j in zip(path,path[1:]):probability*=p[i][j]
    enumeration[path[-1]]+=probability
   assert enumeration==mu;path_checks+=2
  if t in selected:
   history.append(dict(step=t,probabilities=mu,closedFormProbabilityOne=closed,closedFormMatchesEvolution=True,
                       deviationFromStationary=[x-y for x,y in zip(mu,stationary)] if stationary else None,
                       totalVariationFromStationary=tv,worstCaseTotalVariation=worst,
                       forwardFlux=mu[0]*a,backwardFlux=mu[1]*b,netForwardFlux=mu[0]*a-mu[1]*b,
                       cesaroProbabilityOne=average,cesaroDistanceFromStationary=abs(average-stationary[1]) if stationary else None))
  time_checks+=1
  if t!=N:
   power=multiply(power,p);geometric+=lam**t
 next_mu=advance(mu,p);classification['currentDistributionIsStationary']=mu==next_mu
 reverse=[[stationary[j]*p[j][i]/stationary[i] for j in range(2)] if stationary[i]>0 else None for i in range(2)] if stationary else None
 variance=stationary[0]*stationary[1] if stationary else None
 covariance=stationary[1]*p[1][1]-stationary[1]**2 if stationary else None
 if stationary:assert covariance==variance*lam and stationary[0]*a==stationary[1]*b
 mixing=dict(criterion='worst-initial total variation at most 1/4',
             status='not-applicable-nonunique-stationary' if not unique else 'never-periodic' if periodic else
             'not-reached-within-recorded-range' if first_quarter is None else 'reached-within-recorded-range',firstStep=first_quarter)
 reference=dict(transition=p,initial=initial,secondEigenvalue=lam,absoluteContractionCoefficient=abs(lam),
                signedSpectralGap=1-lam,absoluteSpectralGap=1-abs(lam),
                stationaryDistribution=stationary,stationaryFamily='unique-distribution' if unique else 'all-distributions',
                stationaryFamilyDimension=0 if unique else 1,stationarySupport=[i for i,v in enumerate(stationary) if v>0] if stationary else None,
                stationaryBalanceFlow=stationary[0]*a if stationary else None,stationaryTimeReversal=reverse,
                stationaryIndicatorVariance=variance,stationaryLagOneCovariance=covariance,
                stationaryLagOneCorrelation=covariance/variance if variance else None,
                history=history,current=history[-1],nextDistribution=next_mu,cesaroAverage=[1-average,average],
                limitDistribution=limit,cesaroLimitDistribution=stationary if unique else initial,
                mixingAtQuarter=mixing,classification=classification)
 records.append(dict(id=name,parameters=dict(forwardNumerator=A,backwardNumerator=B,transitionDenominator=D,
                                             initialOneNumerator=U,initialDenominator=V,stepCount=N),reference=serialize(reference)))


def count(v):
 if isinstance(v,dict):return 1 if 'numerator'in v else sum(count(x) for x in v.values())
 if isinstance(v,list):return sum(count(x) for x in v)
 return 0


report=dict(scope='independent Fraction matrix powers, path sums, affine closed form and Cesaro geometric sums',cases=records,
            caseCount=len(records),finiteTimeChecks=time_checks,selectedHistoryRows=sum(len(c['reference']['history']) for c in records),
            exactObservationComparisons=sum(count(c['reference']) for c in records),pathEndpointComparisons=path_checks)
Path('scripts/fixtures/two-state-mixing-oracle.json').write_text(json.dumps(report,indent=2)+'\n')
folder=Path('.runtime/learning/two-state-mixing');folder.mkdir(parents=True,exist_ok=True)
(folder/'oracle-verification.json').write_text(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='cases'}))
