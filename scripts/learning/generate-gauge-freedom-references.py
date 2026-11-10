"""Independent spacetime differences, closed-loop quadrature and 180-digit contrasts.
Run: python3 scripts/learning/generate-gauge-freedom-references.py
Requires numpy, scipy and mpmath. No product imports; deterministic exact-input fixtures.
"""
from pathlib import Path
import itertools,json,math
import numpy as np
from scipy.integrate import quad
import mpmath as mp
mp.mp.dps=180
h=1e-3

def first(f,q,i):
 q=np.array(q,dtype=float);e=np.eye(4)[i]*h
 return (f(q-2*e)-8*f(q-e)+8*f(q+e)-f(q+2*e))/(12*h)
def second(f,q,i):
 q=np.array(q,dtype=float);e=np.eye(4)[i]*h
 return (-f(q+2*e)+16*f(q+e)-30*f(q)+16*f(q-e)-f(q-2*e))/(12*h*h)
def curl(A,q):
 d=[first(A,q,i)for i in range(3)];return np.array([d[1][2]-d[2][1],d[2][0]-d[0][2],d[0][1]-d[1][0]])
def div(A,q):return sum(first(A,q,i)[i]for i in range(3))
gauges=[]
for alpha,g2,t,x,y,z,mode in itertools.product([-.8,0.,.7],[-.6,0.,.5],[-.4,.3],[-.7,.4],[-.5,.6],[-.3,.8],['coordinated','vector-only','scalar-only','original']):
 q=np.array([x,y,z,t]);b=lambda t:1+.4*t+.15*t*t;g=lambda t:1-.2*t+.5*g2*t*t;gp=lambda t:-.2+g2*t;useA=mode in['coordinated','vector-only'];useV=mode in['coordinated','scalar-only']
 def A(q):
  x,y,z,t=q;return np.array([-.5*b(t)*y,.5*b(t)*x,0.])+int(useA)*alpha*g(t)*np.array([y*z,x*z,x*y])
 def V(q):x,y,z,t=q;return -int(useV)*alpha*x*y*z*gp(t)
 E=-np.array([first(V,q,i)for i in range(3)])-first(A,q,3);B=curl(A,q);G=div(A,q)+first(V,q,3)
 # Current from independent second derivatives of potentials: curl curl A + grad V_t + A_tt.
 graddiv=np.array([sum(first(lambda q: first(A,q,i)[i],q,j)for i in range(3))for j in range(3)])
 lapA=sum((second(A,q,i)for i in range(3)),np.zeros(3));gradVt=np.array([first(lambda q:first(V,q,3),q,j)for j in range(3)])
 J=graddiv-lapA+gradVt+second(A,q,3)
 corners=[np.array([-.65,-.6,z,t]),np.array([.65,-.6,z,t]),np.array([.65,.6,z,t]),np.array([-.65,.6,z,t])];edges=[]
 for a,end in zip(corners,corners[1:]+corners[:1]):
  d=end-a;edges.append(quad(lambda u:float(A(a+u*d)@d[:3]),0,1,epsabs=1e-12)[0])
 gauges.append(dict(alpha=alpha,g2=g2,mode=mode,query=q.tolist(),electric=E.tolist(),magnetic=B.tolist(),lorenz=G,current=J.tolist(),loopEdges=edges,loopCirculation=sum(edges)))
harmonics=[]
for k,w,v,a,z,t in itertools.product([0.,.6,1.4],[0.,.6,1.4],[.8],[-.5,.7],[.3,.8],[.2]):
 q=np.array([0.,0.,z,t]);phase=lambda q:k*q[2]-w*q[3];V=lambda q:v*math.cos(phase(q));A=lambda q:np.array([0.,0.,a*math.cos(phase(q))])
 rho=-(second(V,q,2)-second(V,q,3));J=-(second(A,q,2)-second(A,q,3))[2]
 E=lambda q:-first(V,q,2)-first(A,q,3)[2];fieldRho=first(E,q,2);fieldJ=-first(E,q,3)
 G=lambda q:first(A,q,2)[2]+first(V,q,3)
 # Direct analytic sources only for differentiating assigned continuity; independent wave source values above.
 D=k*k-w*w;rho_fn=lambda q:D*v*math.cos(phase(q));j_fn=lambda q:D*a*math.cos(phase(q))
 continuity=first(rho_fn,q,3)+first(j_fn,q,2)
 harmonics.append(dict(k=k,omega=w,v=v,a=a,z=z,tau=t,electric=E(q),lorenz=G(q),assignedCharge=rho,assignedCurrent=J,fieldCharge=fieldRho,fieldCurrent=fieldJ,gaussResidual=fieldRho-rho,ampereResidual=fieldJ-J,continuity=continuity))
initial=[]
for a,k,z,t in itertools.product([-.8,0.,.7],[0.,.6,1.4],[-.3,.8],[0.,.3]):
 q=np.array([0.,0.,z,t]);A=lambda q:np.array([0.,0.,a*math.sin(k*q[2])*math.sin(k*q[3])]);E=lambda q:-first(A,q,3)[2]
 G=lambda q:first(A,q,2)[2]
 initial.append(dict(a=a,k=k,z=z,tau=t,electric=E(q),lorenz=G(q),fieldCharge=first(E,q,2),fieldCurrent=-first(E,q,3),vectorWaveResidual=(second(A,q,2)-second(A,q,3))[2],initialG=G(np.array([0.,0.,z,0.])),initialDtG=first(G,np.array([0.,0.,z,0.]),3)))
contrasts=[]
for a,b,c,d in [(1.,1.,1.-2**-52,1.+2**-52),(1.,.7,math.nextafter(1.,math.inf),.7),(1.4,.8,.6,.7),(0.,1.,0.,2.),(1e-100,1e-100,math.nextafter(1e-100,math.inf),1e-100)]:
 exact=mp.mpf(a)*mp.mpf(b)-mp.mpf(c)*mp.mpf(d);contrasts.append(dict(a=a,b=b,c=c,d=d,naive=a*b-c*d,expected=float(exact),exact=str(exact)))
resonances=[]
for k,w in [(1.,math.nextafter(1.,0.)),(1.,math.nextafter(1.,math.inf)),(2.,math.nextafter(2.,0.))]:
 exact=mp.mpf(k)**2-mp.mpf(w)**2;resonances.append(dict(k=k,omega=w,expected=float(exact),exact=str(exact)))
result=dict(provenance='No product imports. Five-point derivatives of scalar/vector potentials, mixed derivatives and wave operators; oriented SciPy edge quadrature; 180-digit exact-input contrasts. Selected numerical observations, not certified global error bounds.',gaugeCases=gauges,harmonicCases=harmonics,initialCases=initial,productContrasts=contrasts,nearResonance=resonances)
p=Path('scripts/fixtures/gauge-freedom-independent.json');p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:len(v)for k,v in result.items()if isinstance(v,list)}))
