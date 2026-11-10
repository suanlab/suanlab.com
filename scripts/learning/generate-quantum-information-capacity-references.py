"""Independent130-decimal Stinespring/Choi eigensystems.
Rates are information values at I/2, promoted to capacities only by the family theorems.
"""
import json
from fractions import Fraction as F
from pathlib import Path
import mpmath as m
m.mp.dps=130
I=m.eye(2);Z=m.matrix([[1,0],[0,-1]]);e0=m.matrix([1,0]);e1=m.matrix([0,1]);bell=m.matrix([1,0,0,1])/m.sqrt(2);bellrho=bell*bell.H
states=[(0,0,0),(0,0,1),(1,0,0),(0,1,0),(0,0,'3/5'),('4/5',0,0),('1/5','2/5','1/5'),('24/25',0,'-7/25')]
states=[tuple(map(F,r)) for r in states]
D=10**30
parameters=list(map(F,['0','1/10','1/4','1/2','3/4','9/10','1']))+[F(D+1,2*D),F(1,D),F(D-1,D)]
def mp(x):return m.mpf(x.numerator)/x.denominator

def kron(a,b):
 return m.matrix([[a[i//b.rows,j//b.cols]*b[i%b.rows,j%b.cols] for j in range(a.cols*b.cols)] for i in range(a.rows*b.rows)])

def matrix(r):
 x,y,z=map(mp,r);return m.matrix([[(1+z)/2,(x-m.j*y)/2],[(x+m.j*y)/2,(1-z)/2]])

def entropy(a):
 eigenvalues,_=m.eighe(a)
 return -sum(v*m.log(v,2) for v in eigenvalues if v>m.mpf('1e-120'))

def scalar(x):return m.nstr(m.re(x),120)
def matrix_wire(a):return [[dict(real=scalar(a[i,j]),imaginary=scalar(m.im(a[i,j]))) for j in range(a.cols)] for i in range(a.rows)]
def outputs(p,rho):
 W=m.sqrt(1-p)*kron(I,e0)+m.sqrt(p)*kron(Z,e1)
 joint=W*rho*W.H
 B=m.matrix([[sum(joint[2*i+k,2*j+k] for k in range(2)) for j in range(2)] for i in range(2)])
 E=m.matrix([[sum(joint[2*k+i,2*k+j] for k in range(2)) for j in range(2)] for i in range(2)])
 assert m.norm(W.H*W-I)<m.mpf('1e-120')
 assert abs(sum(B[i,i] for i in range(2))-1)<m.mpf('1e-120')
 assert abs(sum(E[i,i] for i in range(2))-1)<m.mpf('1e-120')
 return B,E
cases=[]
for fraction in parameters:
 p=mp(fraction);Bopt,Eopt=outputs(p,I/2);Q=entropy(Bopt)-entropy(Eopt);CE=1+Q
 choi=(1-p)*bellrho+p*kron(I,Z)*bellrho*kron(I,Z).H
 for index,r in enumerate(states):
  rho=matrix(r);B,E=outputs(p,rho);sa,sb,se=entropy(rho),entropy(B),entropy(E)
  cases.append(dict(name=f'{fraction}/{index}',p=str(fraction),bloch=list(map(str,r)),outputDensity=matrix_wire(B),environmentDensity=matrix_wire(E),normalizedChoi=matrix_wire(choi),
   expected=dict(inputEntropy=scalar(sa),outputEntropy=scalar(sb),environmentEntropy=scalar(se),selectedInputCoherentInformation=scalar(sb-se),selectedInputReferenceMutualInformation=scalar(sa+sb-se)),quantumRate=scalar(Q),assistedRate=scalar(CE)))
output=Path('scripts/fixtures/quantum-information-capacity-independent.json')
output.write_text(json.dumps(dict(method='mpmath Stinespring output partial traces and Hermitian matrix eigensystems at130 decimal digits',comparisonAllowance='1e-110',cases=cases),indent=2)+'\n')
print(json.dumps(dict(cases=len(cases),file=str(output))))
