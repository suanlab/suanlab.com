"""Finite physical-volume semigroups and exact nonlinear entropy budgets.

Independent Taylor/Poisson evolutions and rational flux quadratures supplement
the analytic arguments; these finite cases are not a general solver certificate.
"""
from fractions import Fraction as F
from math import exp,log,isclose
import json

def mv(M,x):return [sum(a*b for a,b in zip(row,x)) for row in M]
def setup(V,edges):
 n=len(V);Q=[[F(0) for _ in V] for _ in V]
 for i,g in enumerate(edges):
  j=(i+1)%n;Q[i][j]+=g/V[i];Q[j][i]+=g/V[j];Q[i][i]-=g/V[i];Q[j][j]-=g/V[j]
 return Q
cases=0
for V,edges,c0 in [([F(1),F(1),F(1)],[F(1)]*3,[1.,0.,0.]),([F(1),F(2),F(1)],[F(1)]*3,[1.,0.,3.]),([F(1),F(3),F(2),F(4)],[F(1),F(0),F(1),F(0)],[1.,0.,0.,0.]),([F(1),F(3),F(2),F(4)],[F(0)]*4,[1.,0.,3.,2.]),([F(1),F(2),F(1)],[F(1)]*3,[0.,0.,0.])]:
 Q=setup(V,edges);n=len(V);pi=[v/sum(V) for v in V];rate=max(-Q[i][i] for i in range(n));assert all(sum(row)==0 for row in Q)
 for i in range(n):
  assert sum(pi[j]*Q[j][i] for j in range(n))==0
  for j in range(n):assert pi[i]*Q[i][j]==pi[j]*Q[j][i]
 if rate:
  P=[[F(int(i==j))+Q[i][j]/rate for j in range(n)] for i in range(n)];assert all(min(row)>=0 and sum(row)==1 for row in P)
 mass=float(sum(v*c for v,c in zip(V,c0)));mean=mass/float(sum(V))
 def entropy(c):return sum(float(v)*x*log(x/mean) if x>0 else 0. for v,x in zip(V,c))
 for t in [.01,.5,2.]:
  if rate:
   w=exp(-float(rate)*t);term=list(c0);result=[w*x for x in term]
   for j in range(1,180):
    term=mv(P,term);w*=float(rate)*t/j;result=[a+w*b for a,b in zip(result,term)]
  else:result=list(c0)
  # A separate Taylor series uses Q, not the stochastic matrix P.
  term=list(c0);reference=list(c0)
  for j in range(1,180):
   term=[t*x/j for x in mv(Q,term)];reference=[a+b for a,b in zip(reference,term)]
   if max(map(abs,term))<1e-18:break
  assert max(abs(a-b) for a,b in zip(result,reference))<2e-11
  assert min(result)>=0 and isclose(sum(float(v)*x for v,x in zip(V,result)),mass,rel_tol=1e-13,abs_tol=1e-13)
  if mass>0:assert entropy(result)<=entropy(c0)+2e-12
  else:assert result==c0==[0.,0.,0.]
  for power in [2,4]:assert sum(float(v)*x**power for v,x in zip(V,result))<=sum(float(v)*x**power for v,x in zip(V,c0))+2e-12
  if not rate:assert result==c0
  if V==[F(1),F(1),F(1)]:assert min(result)>0 and max(abs(x-(1/3+(c0[i]-1/3)*exp(-3*t))) for i,x in enumerate(result))<2e-13
  if edges==[F(1),F(0),F(1),F(0)]:assert result[2:]==[0.,0.]
  cases+=1

# Exact rational flux, face entropy and time-step budgets, independently assembled.
def fec(a,b):return (a*a+a*b+b*b)/6
def flux(a,b,alpha):return fec(a,b)-alpha*(b-a)/2
face=0
for a in [F(-2),F(-1,2),F(0),F(1),F(2)]:
 for b in [F(-2),F(-1,2),F(0),F(1),F(2)]:
  assert (b-a)*fec(a,b)==(b**3-a**3)/6
  for alpha in [F(0),F(1,3),F(2)]:
   assert (b**3-a**3)/6-(b-a)*flux(a,b,alpha)==alpha*(b-a)**2/2>=0
   face+=1
periodic=0
for q in [[F(0),F(2),F(1)],[F(-1),F(1),F(0)],[F(1),F(1),F(1)],[F(1,4),F(-1,2),F(2),F(0)]]:
 for alpha in [F(0),F(1,3),F(2)]:
  for dt in [F(1,100),F(1,10),F(7,10)]:
   h=[F(i+1,2) for i in range(len(q))]
   f=[flux(a,b,alpha) for a,b in zip(q,q[1:]+q[:1])]
   rate=[(f[i-1]-f[i])/h[i] for i in range(len(q))]
   updated=[a+dt*b for a,b in zip(q,rate)]
   defect=[(b**3-a**3)/6-(b-a)*ff for a,b,ff in zip(q,q[1:]+q[:1],f)]
   E=lambda state:sum(v*x*x for v,x in zip(h,state))/2
   assert sum(v*x for v,x in zip(h,updated))==sum(v*x for v,x in zip(h,q))
   assert sum(v*x*r for v,x,r in zip(h,q,rate))==-sum(defect)<=0
   assert E(updated)-E(q)==-dt*sum(defect)+dt*dt*sum(v*r*r for v,r in zip(h,rate))/2
   periodic+=1
q=[F(0),F(1),F(0)];dt=F(3,5);updated=[dt,1-2*dt,dt]
assert sum(updated)==1 and min(updated)==F(-1,5)
assert sum(x*x for x in updated)/2==F(19,50)<F(1,2) and sum(abs(x) for x in updated)==F(7,5)>1
q=[F(0),F(2),F(1)];fa=[(a*a+b*b)/4 for a,b in zip(q,q[1:]+q[:1])];r=[fa[i-1]-fa[i] for i in range(3)];assert sum(x*y for x,y in zip(q,r))==F(1,2)>0
print(json.dumps(dict(weightedSemigroupAndConvexEntropyCases=cases,includesDisconnectedAndZeroGenerator=True,independentTaylorAndUniformization=True,exactFaceEntropyCases=face,exactNonuniformPeriodicEulerBudgets=periodic,exactNormPositivityCounterexamples=1,exactArithmeticFluxEntropyCounterexamples=1),indent=2))
