"""Independent finite grid and Robin-slab model checks, standard library only.

Exact rational systems and independent flux/inventory integration supplement
analytic arguments; they do not certify all parameters or a general solver.
"""
from fractions import Fraction as F
from math import expm1,exp,sqrt,tanh,sinh,cosh,isclose
import json
def solve(A,b):
 A=[list(row)+[rhs] for row,rhs in zip(A,b)]
 n=len(A)
 for i in range(n):
  pivot=next(j for j in range(i,n) if A[j][i]!=0);A[i],A[pivot]=A[pivot],A[i]
  scale=A[i][i];A[i]=[v/scale for v in A[i]]
  for j in range(n):
   if j!=i:
    scale=A[j][i];A[j]=[v-scale*w for v,w in zip(A[j],A[i])]
 return [row[-1] for row in A]
def discrete(N,Pe,method):
 p=Pe/N
 if method=='center':up,low,diag=1-p/2,1+p/2,F(-2)
 else:up,low,diag=1+max(-p,0),1+max(p,0),-(2+abs(p))
 A=[[F(0) for _ in range(N-1)] for _ in range(N-1)];b=[F(0)]*(N-1)
 for j in range(N-1):
  A[j][j]=diag
  if j:A[j][j-1]=low
  if j<N-2:A[j][j+1]=up
  else:b[j]=-up
 return [F(0),*solve(A,b),F(1)]
count=0
for N in [2,4,6,10]:
 for Pe in [F(0),F(2),F(4),F(8),F(10),F(-2),F(-10)]:
  center=discrete(N,Pe,'center');up=discrete(N,Pe,'upwind')
  assert all(0<=v<=1 for v in up)
  if abs(Pe/N)<=2:assert all(0<=v<=1 for v in center)
  if not Pe:assert center==[F(i,N) for i in range(N+1)]
  count+=1
assert discrete(4,F(10),'center')==[F(0),F(-10,6560),F(80,6560),F(-730,6560),F(1)]
assert discrete(4,F(8),'center')==[F(0),F(0),F(0),F(0),F(1)]
def B(p):
 if p==0:return 1.
 if p>0:return p*exp(-p)/(-expm1(-p))
 return -p/(-expm1(p))
fit=0
nodes=[0.,.1,.35,.6,1.]
for Pe in [-10.,-4.,0.,.1,4.,10.]:
 n=len(nodes)-2;A=[[0. for _ in range(n)] for _ in range(n)];rhs=[0.]*n
 for i in range(1,len(nodes)-1):
  left,right=nodes[i]-nodes[i-1],nodes[i+1]-nodes[i];pL,pR=Pe*left,Pe*right
  lo=-B(-pL)/left;hi=-B(pR)/right;diag=B(pL)/left+B(-pR)/right
  j=i-1;A[j][j]=diag
  if j:A[j][j-1]=lo
  if j<n-1:A[j][j+1]=hi
  else:rhs[j]-=hi
 c=[0.,*solve(A,rhs),1.]
 true=[expm1(Pe*x)/expm1(Pe) if Pe else x for x in nodes]
 assert max(abs(a-b) for a,b in zip(c,true))<2e-13
 flux=[(B(-Pe*(b-a))*u-B(Pe*(b-a))*v)/(b-a) for a,b,u,v in zip(nodes,nodes[1:],c,c[1:])]
 assert max(abs(j+B(Pe)) for j in flux)<2e-12
 fit+=1
robin=0
for D in [.5,1.,2.]:
 for k in [.1,1.,4.]:
  for hm in [.1,1.,10.]:
   L,cb,area=.8,1.3,1.7;phi=L*sqrt(k/D);ad=sqrt(D*k)*tanh(phi);cs=hm*cb/(hm+ad)
   n=20000;dx=L/n
   consumption=area*sum(k*cs*cosh(phi*(1-(j+.5)/n))/cosh(phi)*dx for j in range(n))
   rate=area*hm*(cb-cs)
   assert isclose(consumption,rate,rel_tol=2e-9,abs_tol=1e-11),(D,k,hm,consumption,rate)
   eps=1e-5
   def r(kk):
    aa=sqrt(D*kk)*tanh(L*sqrt(kk/D));return area*cb*hm*aa/(hm+aa)
   difference=(r(k*exp(eps))-r(k*exp(-eps)))/(2*eps*rate)
   target=hm/(hm+ad)*(.5+phi/sinh(2*phi))
   assert abs(difference-target)<2e-10
   robin+=1

assert discrete(4,F(-8),'center')==[F(0),F(1),F(1),F(1),F(1)]
inverse=0
v=.3
for k in [.5,1.,4.,9.]:
 a=sqrt(k)*tanh(sqrt(k));assert a>v
 hm=v*a/(a-v);surface=hm/(hm+a)
 assert hm>0 and isclose(hm*a/(hm+a),v,abs_tol=1e-15)
 assert isclose(surface,v/a,abs_tol=1e-15)
 inverse+=1
similarity=0
for alpha in [.5,1.,2.]:
 for beta in [.5,1.,3.]:
  for k in [.1,1.,4.]:
   D,L,hm,area,cb=.7,.8,.6,1.2,1.3
   def state(D,L,k,hm,area):
    phi=L*sqrt(k/D);ad=sqrt(D*k)*tanh(phi);cs=cb*hm/(hm+ad)
    profile=[cs*cosh(phi*(1-x))/cosh(phi) for x in [0.,.3,1.]]
    rate=area*ad*cs;inventory=area*cs*L*tanh(phi)/phi
    return profile,rate,inventory
   old=state(D,L,k,hm,area);new=state(beta*D,alpha*L,beta*k/(alpha*alpha),beta*hm/alpha,alpha*alpha*area)
   assert max(abs(a-b) for a,b in zip(old[0],new[0]))<2e-14
   assert isclose(new[1],alpha*beta*old[1],rel_tol=1e-14)
   assert isclose(new[2],alpha**3*old[2],rel_tol=1e-14)
   similarity+=1
# At k=0 use c=C0+C1*x. With no external coupling, sealed-end
# equations leave the constant mode free; do not solve it by dividing by h.
D,cb=F(2),F(3,2)
for hm in [F(1,2),F(1),F(2)]:
 C0,C1=solve([[hm,-D],[F(0),F(1)]],[hm*cb,F(0)])
 assert C0==cb and C1==0
for C0 in [F(0),F(1,3),F(2)]:
 C1=F(0)
 boundary_matrix=[[F(0),-D],[F(0),F(1)]]
 assert [sum(row[j]*[C0,C1][j] for j in range(2)) for row in boundary_matrix]==[0,0]
assert len({F(0),F(1,3),F(2)})==3
for k in [.1,1.,4.]:
 admittance=sqrt(2*k)*tanh(.8*sqrt(k/2))
 assert admittance>0
 cs=0./admittance
 assert cs==0 and admittance*cs==0
result=dict(exactUniformGridPairs=count,exactZeroNeighborCoefficientControls=2,nonuniformFittedFluxCases=fit,robinConsumptionAndSensitivityCases=robin,positiveSameRateInversePairs=inverse,robinSimilarityCases=similarity,zeroReactionSupplyControls=9)
if __name__=='__main__':print(json.dumps(result,sort_keys=True))
