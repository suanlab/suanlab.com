"""Stipulated rational log-two preparations; NOT an actual-binary64 oracle."""
from fractions import Fraction as F
from pathlib import Path
import json
spins=[(-1,-1),(-1,1),(1,-1),(1,1)]
pi=[F(2,5),F(1,10),F(1,10),F(2,5)]
def kernel(site,states,prob):
 rows=[]
 for x in states:
  eligible=[j for j,y in enumerate(states) if y[1-site]==x[1-site]]
  denom=sum(prob[j] for j in eligible)
  rows.append([prob[j]/denom if j in eligible else F(0) for j in range(len(states))])
 return rows
def mm(A,B):return [[sum(a*b for a,b in zip(row,col)) for col in zip(*B)] for row in A]
def stationary(p,K):return [sum(p[i]*K[i][j] for i in range(len(p))) for j in range(len(p))]
def db(p,K):return [[p[i]*K[i][j]-p[j]*K[j][i] for j in range(len(p))] for i in range(len(p))]
K1,K2=[kernel(i,spins,pi) for i in [0,1]]
random=[[ (a+b)/2 for a,b in zip(r,s)] for r,s in zip(K1,K2)];systematic=mm(K1,K2)
for K in [K1,K2,random,systematic]:
 assert all(sum(row)==1 and all(p>=0 for p in row) for row in K)
 assert stationary(pi,K)==pi
assert all(v==0 for row in db(pi,random) for v in row)
assert pi[0]*systematic[0][1]==F(8,125) and pi[1]*systematic[1][0]==F(2,125)
restricted=[(-1,1),(1,-1)];p0=[F(1,2),F(1,2)]
identity=[[F(1),F(0)],[F(0),F(1)]];block=[p0,p0]
assert kernel(0,restricted,p0)==kernel(1,restricted,p0)==identity
assert stationary([F(1),F(0)],identity)==[F(1),F(0)]
assert stationary([F(1),F(0)],block)==p0
weights=[F(1),F(1),F(1,4)];partition=sum(weights);p=[w/partition for w in weights]
assert partition==F(9,4) and p==[F(4,9),F(4,9),F(1,9)]
assert sum(p[i]*i for i in range(3))==F(2,3)
assert sum(p[i]*i*i for i in range(3))-F(2,3)**2==F(4,9)
precision=[[F(2),F(1)],[F(1),F(2)]];cov=[[F(2,3),F(-1,3)],[F(-1,3),F(2,3)]]
assert mm(precision,cov)==identity
mu=[cov[0][0],cov[1][0]];assert mu==[F(2,3),F(-1,3)]
minimum=-mu[0]/2;assert minimum==F(-1,3) and minimum+1==F(2,3)
def serialize(x):
 if isinstance(x,F):return str(x)
 if isinstance(x,list):return [serialize(v) for v in x]
 if isinstance(x,dict):return {k:serialize(v) for k,v in x.items()}
 return x
out=dict(scope='Ideal rational examples with stipulated beta*J=ln2 or beta*epsilon=ln2. No actual binary64 reference or new model/reader completion claimed.',weighted=dict(weights=weights,Z=partition,probabilities=p,ZnormalizedPrior=partition/4,meanIndex=F(2,3),indexVariance=F(4,9)),gibbs=dict(states=spins,target=pi,K1=K1,K2=K2,randomScan=random,systematicSweep=systematic,randomDetailedBalanceDefect=db(pi,random),systematicDetailedBalanceDefect=db(pi,systematic),restrictedSingleSite=identity,restrictedBlock=block),gaussian=dict(commonOffset=F(0),dimensionlessPrecision=precision,covariance=cov,shiftedMean=mu,shiftedMinimumEnergyInThermalUnits=minimum,shiftedMeanEnergyInThermalUnits=minimum+1,heatCapacityInKB=F(1),partitionAtZeroSource='2*pi/sqrt(3)'))
Path('scripts/fixtures/boltzmann-gibbs-ideal-hand-cases.json').write_text(json.dumps(serialize(out),indent=2)+'\n')
print(json.dumps(dict(weightedNormalization=True,stochasticKernels=4,stationaryKernels=4,randomScanDetailedBalance=True,systematicNonreversible=True,restrictedSupportCounterexample=True,gaussianInverseAndMoments=True,actualInputOracle=False)))
