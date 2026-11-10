"""Exact independent identities for the next correlation topic; not model evidence."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import json
pairs=[]
for sign in [-1,1]:
 states=list(product([-1,1],repeat=2));raw=[F(3) if s[0]*s[1]*sign>0 else F(1) for s in states];Z=sum(raw);p=[w/Z for w in raw]
 means=[sum(w*s[i] for s,w in zip(states,p)) for i in range(2)]
 C=[[sum(w*(s[i]-means[i])*(s[j]-means[j]) for s,w in zip(states,p)) for j in range(2)] for i in range(2)]
 assert means==[0,0] and C==[[F(1),F(sign,2)],[F(sign,2),F(1)]]
 assert C[0][0]*C[1][1]-C[0][1]*C[1][0]==F(3,4)
 susceptibility=sum(sum(row) for row in C)/2
 assert susceptibility==1+F(sign,2)
 pairs.append(dict(idealK='ln(3)/2' if sign>0 else '-ln(3)/2',covariance=[[str(v) for v in row] for row in C],uniformSusceptibilityPerSpin=str(susceptibility)))
rings=[]
for N in range(3,9):
 for sign in [-1,1]:
  t=F(sign,2);states=list(product([-1,1],repeat=N));raw=[]
  for state in states:
   walls=sum(state[i]!=state[(i+1)%N] for i in range(N));raw.append(F(3)**(-walls if sign>0 else walls))
  Z=sum(raw);p=[w/Z for w in raw]
  correlations=[sum(w*s[0]*s[r] for s,w in zip(states,p)) for r in range(N)]
  formula=[(t**r+t**(N-r))/(1+t**N) for r in range(N)]
  assert correlations==formula
  response=sum(correlations)
  assert response==(1+t)/(1-t)*(1-t**N)/(1+t**N)>0
  varM=sum(w*F(sum(s),N)**2 for s,w in zip(states,p))
  assert response==N*varM
  rings.append(dict(N=N,t=str(t),enumeratedConfigurations=len(states),correlations=[str(v) for v in correlations],response=str(response)))
ar=[]
for rho in [F(-1),F(-1,2),F(0),F(1,2),F(1)]:
 for n in range(1,9):
  covariance=[[rho**abs(i-j) for j in range(n)] for i in range(n)]
  direct=sum(sum(row) for row in covariance)/(n*n)
  weighted=(n+2*sum((n-k)*rho**k for k in range(1,n)))/(n*n)
  sums=[sum(rho**k for k in range(m)) for m in range(1,n+1)]
  positive=(sums[-1]**2+(1-rho*rho)*sum(v*v for v in sums[:-1]))/(n*n)
  assert direct==weighted==positive and direct>=0
  if rho==1:assert direct==1
  if rho==-1:assert direct==(F(0) if n%2==0 else F(1,n*n))
  ar.append(dict(rho=str(rho),n=n,meanVariance=str(direct),varianceEquivalentCount='positive-infinity' if not direct else str(1/direct)))
result=dict(scope='Exact stipulated rational hand identities only; actual binary64 models, manuscripts and readers are not implemented by this script',binaryPairs=pairs,periodicRings=rings,ar1FiniteMeans=ar,ringEnumerationCount=sum(x['enumeratedConfigurations'] for x in rings))
out=Path('.runtime/learning/phase-transitions-integration/next-correlations-hand-checks.json');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(binaryCases=len(pairs),ringCases=len(rings),enumeratedRingConfigurations=result['ringEnumerationCount'],ar1Cases=len(ar),referenceOnly=True)))
