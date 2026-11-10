"""Independent expm/SVD, DOP853 and polynomial/Fraction references.
Run from repository root with numpy, scipy and sympy available.
"""
from pathlib import Path
from fractions import Fraction as Q
from math import comb,sqrt
import json,random
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
import sympy as sy
rng=random.Random(20261104420)
# Power balance p-2=p/2, then coefficient balance 12*A=12*c*sqrt(A).
p=sy.Symbol('p');assert sy.solve(p-2-p/2,p)==[4]
a,c=sy.symbols('a c',positive=True);assert sy.solve(12*a*a-12*c*a,a)==[c]
def label(x):return str(x)
def energy(m,k,x,v):return Q(m)*Q(v,10)**2/2-Q(k)*Q(x,10)**2/2
def linear(values):
 m,k,xa,va,xb,vb,tc=values;t=tc/10;A=np.array([[0.,1.],[k/m,0.]])
 M=expm(t*A);za=np.array([xa,va])/10;zb=np.array([xb,vb])/10;qa=M@za;qb=M@zb;sig=np.linalg.svd(M,compute_uv=False)
 return dict(values=values,matrix=M.tolist(),queryA=qa.tolist(),queryB=qb.tolist(),sigma=sig.tolist(),energyA=label(energy(m,k,xa,va)),energyB=label(energy(m,k,xb,vb)),delta=(M@(zb-za)).tolist())
linear_inputs=[[m,k,x,v,-x,-v,t]for m in range(1,5)for k in range(5)for x,v in [(0,0),(10,10),(10,-10),(-10,10),(20,-20)]for t in [0,1,10,50]]
# 400 systematic +400 random, including all free/mass/stiffness extrema.
assert len(linear_inputs)==400
linear_inputs += [[rng.randint(1,4),rng.randint(0,4),*[rng.randint(-20,20)for _ in range(4)],rng.randint(0,50)]for _ in range(400)]
lin=[linear(v)for v in linear_inputs]
linear_ode=[]
for j,v in enumerate(linear_inputs[::5]):
 m,k,xa,va,xb,vb,tc=v;t=tc/10
 if t==0:out=np.array([xa,va,xb,vb],dtype=float)/10
 else:
  sol=solve_ivp(lambda t,z:[z[1],k/m*z[0],z[3],k/m*z[2]],(0,t),np.array([xa,va,xb,vb])/10,method='DOP853',rtol=2e-13,atol=2e-14);assert sol.success;out=sol.y[:,-1]
 linear_ode.append(dict(values=v,state=out.tolist()))
def eval_poly(coeff,t,derivative=0):
 for _ in range(derivative):coeff=[j*x for j,x in enumerate(coeff)][1:]
 return sum((a*t**j for j,a in enumerate(coeff)),Q(0))
def released(m,c,tau,t):
 if t<=tau:coeff=[Q(0)]*5
 else:coeff=[c*c*comb(4,j)*(-tau)**(4-j)for j in range(5)]
 x=eval_poly(coeff,t);v=eval_poly(coeff,t,1);acc=eval_poly(coeff,t,2)
 root=c*(t-tau)**2 if t>tau else Q(0)
 K=Q(m)*v*v/2;U=-8*m*c*root**3
 assert x>=0 and root**2==x and Q(m)*acc==12*m*c*root and K+U==0
 return {key:label(value)for key,value in dict(position=x,velocity=v,acceleration=acc,force=Q(m)*acc,kinetic=K,potential=U,energy=K+U).items()}
delayed_inputs=[[m,c,a,b,t]for m in [1,4]for c in [0,1,2,4]for a,b in [(0,0),(0,10),(10,0),(10,50),(50,50)]for t in [0,1,10,50,100]]
assert len(delayed_inputs)==200
delayed_inputs += [[rng.randint(1,4),rng.randint(0,4),rng.randint(0,50),rng.randint(0,50),rng.randint(0,100)]for _ in range(500)]
delayed=[]
for m,cc,ac,bc,tc in delayed_inputs:
 c=Q(cc,2);t=Q(tc,10);a=Q(ac,10);b=Q(bc,10)
 delayed.append(dict(values=[m,cc,ac,bc,tc],queryA=released(m,c,a,t),queryB=released(m,c,b,t)))
post_ode=[]
for j in range(180):
 m=rng.randint(1,4);cc=rng.randint(1,4);c=cc/2;start=rng.randint(1,10)/10;finish=start+rng.randint(1,50)/10
 initial=released(m,Q(cc,2),Q(0),Q(str(start)));y0=[float(Q(initial[k]))for k in ['position','velocity']]
 sol=solve_ivp(lambda t,z:[z[1],12*c*sqrt(max(z[0],0))],(start,finish),y0,method='DOP853',rtol=2e-13,atol=2e-14);assert sol.success
 expected=released(m,Q(cc,2),Q(0),Q(str(finish)));err=max(abs(sol.y[i,-1]-float(Q(expected[k])))/max(1,abs(float(Q(expected[k]))))for i,k in enumerate(['position','velocity']));assert err<2e-10,err
 post_ode.append(dict(mass=m,coefficientCode=cc,start=start,finish=finish,state=sol.y[:,-1].tolist(),expected=expected,scaledError=err))
result=dict(linear=lin,linearODE=linear_ode,delayed=delayed,postReleaseODE=post_ode)
Path('scripts/fixtures/initial-state-role-independent.json').write_text(json.dumps(result,indent=2)+'\n')
print('Generated',len(lin),'expm/SVD;',len(linear_ode),'linear ODE;',len(delayed),'exact polynomial;',len(post_ode),'positive-state release ODE references; max postrelease scaled error',max(x['scaledError']for x in post_ode))
