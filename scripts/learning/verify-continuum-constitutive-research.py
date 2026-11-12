"""Independent finite constitutive-model checks; standard library only.

Exact rational algebra, actual matrix-energy differences, convolution and
power integration supplement the manuscript arguments, not global stability.
"""
from fractions import Fraction as F
from itertools import product
import json
def mod(E,tau,w):
 x=tau*w
 return E*x*x/(1+x*x),E*x/(1+x*x)
fit=second=0
for a,b,w in product([F(1,2),F(1),F(2)],repeat=3):
 E=(a*a+b*b)/a;tau=a/(b*w);assert mod(E,tau,w)==(a,b)
 r0=a/b;x1=r0/4;x2=4*r0
 weights=[b*(x2-r0)/(x2-x1),b*(r0-x1)/(x2-x1)]
 branches=[(weight*(1+x*x)/x,x/w) for weight,x in zip(weights,[x1,x2])]
 assert all(e>0 and t>0 for e,t in branches)
 measured=[sum(mod(e,t,w)[j] for e,t in branches) for j in [0,1]]
 assert measured==[a,b];fit+=1
 for r in [F(1,4),F(1,2),F(1),F(2)]:
  maxwell=mod(E,tau,r*w);kv=(a,r*b)
  assert (maxwell==kv)==(r==1);second+=1
I=[[F(int(i==j)) for j in range(3)] for i in range(3)]
R=[[F(3,5),F(-4,5),F(0)],[F(4,5),F(3,5),F(0)],[F(0),F(0),F(1)]]
def tr(A):return sum(A[i][i] for i in range(3))
def transpose(A):return [list(x) for x in zip(*A)]
def mul(A,B):return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
def cofactor(A):
 out=[]
 for i in range(3):
  row=[]
  for j in range(3):
   B=[[A[k][l] for l in range(3) if l!=j] for k in range(3) if k!=i]
   row.append((-1)**(i+j)*(B[0][0]*B[1][1]-B[0][1]*B[1][0]))
  out.append(row)
 return out
def det(A):return sum(A[0][j]*cofactor(A)[0][j] for j in range(3))
def invariants(A):
 C=mul(transpose(A),A);i1=tr(C);i2=(i1*i1-tr(mul(C,C)))/2;i3=det(C)
 disc=i1*i1*i2*i2-4*i2**3-4*i1**3*i3+18*i1*i2*i3-27*i3*i3
 d1=2*i1*i2*i2-12*i1*i1*i3+18*i2*i3
 d2=2*i1*i1*i2-12*i2*i2+18*i1*i3
 d3=-4*i1**3+18*i1*i2-54*i3
 cof=cofactor(C)
 gradC=[[d1*I[i][j]+d2*(i1*I[i][j]-C[i][j])+d3*cof[i][j] for j in range(3)] for i in range(3)]
 gradF=[[2*x for x in row] for row in mul(A,gradC)]
 return (i1,i2,i3,disc),gradF
uniaxial=0
for scale,L,Q in product([F(1,2),F(1),F(3,2),F(2)],[I,R],[I,R]):
 A=[[F(0) for _ in range(3)] for _ in range(3)];A[0][0]=scale*scale;A[1][1]=A[2][2]=1/scale
 A=mul(mul(L,A),Q);v,g=invariants(A);assert det(A)==1 and v[3]==0
 assert all(x==0 for row in g for x in row);uniaxial+=1
shear=0
for gamma,kap in product([F(-2),F(-1,2),F(0),F(1,4),F(1,2),F(1),F(3,2)],[F(0),F(1,100),F(1,10)]):
 A=[row[:] for row in I];A[0][1]=gamma;v,g=invariants(A)
 assert v[0]==v[1]==3+gamma*gamma and v[2]==1
 assert v[3]==gamma**6*(gamma*gamma+4)>=0
 assert gamma+kap*g[0][1]==gamma+8*kap*gamma**5*(gamma*gamma+3)
 for L,Q in product([I,R],repeat=2):
  vv,gg=invariants(mul(mul(L,A),Q));assert vv==v
  assert gg==mul(mul(L,g),Q)
 shear+=1

# Independent scalar-energy central differences on unconstrained matrix
# directions, keeping the full determinant invariant.
differences=0
for gamma in [F(1,2),F(1),F(3,2)]:
 A=[row[:] for row in I];A[0][1]=gamma;A=mul(R,A)
 _,g=invariants(A)
 for i,j in product(range(3),repeat=2):
  errors=[]
  for step in [F(1,10000),F(1,20000)]:
   ap=[row[:] for row in A];am=[row[:] for row in A]
   ap[i][j]+=step;am[i][j]-=step
   difference=(invariants(ap)[0][3]-invariants(am)[0][3])/(2*step)
   errors.append(abs(difference-g[i][j]))
  assert errors[1]<=errors[0]*F(1,3)
  differences+=1
# Omitting I3 derivatives at fixed determinant changes the normal extension.
# A distinct axisymmetric determinant-one state with rational entries:
A=[[F(4),F(0),F(0)],[F(0),F(1,2),F(0)],[F(0),F(0),F(1,2)]]
(v1,v2,v3,dd),gradient=invariants(A);assert dd==0
c=mul(transpose(A),A);d1=2*v1*v2*v2-12*v1*v1*v3+18*v2*v3;d2=2*v1*v1*v2-12*v2*v2+18*v1*v3
reduced=[[d1*I[i][j]+d2*(v1*I[i][j]-c[i][j]) for j in range(3)] for i in range(3)]
d3=-4*v1**3+18*v1*v2-54*v3
assert reduced==[[-d3*x for x in row] for row in cofactor(c)]
assert any(x!=0 for row in reduced for x in row)
assert all(x==0 for row in gradient for x in row)

from math import exp,sin,cos,pi,isclose
branches=[(3.4,.25),(.85,4.)]
convolution=0
amplitude=.002
for E,tau in branches:
 x=tau;storage=E*x*x/(1+x*x);loss=E*x/(1+x*x)
 for t in [0.,.2,1.,3.,2*pi]:
  n=24000;step=t/n
  integrated=sum(exp(-(t-(j+.5)*step)/tau)*amplitude*cos((j+.5)*step)*step for j in range(n))
  actual=amplitude*loss*exp(-t/tau)+E*integrated
  periodic=amplitude*(storage*sin(t)+loss*cos(t))
  assert isclose(actual,periodic,abs_tol=3e-11,rel_tol=2e-7)
  convolution+=1

def state(model,t,amp):
 strain=amp*sin(t);rate=amp*cos(t)
 if model=='KV':return strain+rate,strain*strain/2,rate*rate
 elements=[(2.,1.)] if model=='M' else branches
 stress=energy=dissipation=0.
 for E,tau in elements:
  x=tau;sig=amp*(E*x*x/(1+x*x)*sin(t)+E*x/(1+x*x)*cos(t))
  stress+=sig;energy+=sig*sig/(2*E);dissipation+=sig*sig/(E*tau)
 return stress,energy,dissipation
power_cases=cycles=0
for model in ['KV','M','2M']:
 for t in [0.,.2,1.,3.,2*pi]:
  stress,energy,diss=state(model,t,amplitude);step=1e-5
  derivative=(state(model,t+step,amplitude)[1]-state(model,t-step,amplitude)[1])/(2*step)
  assert diss>=0 and abs(stress*amplitude*cos(t)-derivative-diss)<1e-14
  assert isclose(stress,amplitude*(sin(t)+cos(t)),abs_tol=1e-18)
  power_cases+=1
 for amp in [-.002,.002]:
  n=24000;step=2*pi/n;work=loss=0.
  for j in range(n):
   t=(j+.5)*step;sig,energy,diss=state(model,t,amp)
   work+=sig*amp*cos(t)*step;loss+=diss*step
  assert isclose(work,pi*amp*amp,abs_tol=1e-16)
  assert isclose(loss,work,abs_tol=1e-16)
  assert isclose(state(model,0,amp)[1],state(model,2*pi,amp)[1],abs_tol=1e-18)
  cycles+=1
# Same stress at the sine origin, different exact internal storage.
storage_states=[F(0),mod(F(2),F(1),F(1))[1]**2/4,sum(mod(e,t,F(1))[1]**2/(2*e) for e,t in [(F(17,5),F(1,4)),(F(17,20),F(4))])]
assert storage_states==[F(0),F(1,4),F(2,17)] and len(set(storage_states))==3
result=dict(exactDistinctPeriodicInitialStorage=1,exactPositiveInverseSpectrumFits=fit,exactOtherFrequencyComparisons=second,exactRotatedUniaxialInvisibleGradients=uniaxial,exactShearObjectiveGradientCases=shear,independentMatrixEnergyGradientDifferences=differences,reducedConstraintGradientNegativeControl=1,independentMemoryConvolutions=convolution,independentPeriodicPowerBalances=power_cases,independentCycleWorkAndLossIntegrals=cycles)
if __name__=='__main__':print(json.dumps(result,sort_keys=True))
