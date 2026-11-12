#!/usr/bin/env python3
"""Independent exact graph algebra / constraint balances. No product imports.
Usage: python3 scripts/learning/verify-gauge-research.py --json|--references
Finite verification complements the manuscript proofs; it is not a solver certificate.
"""
from fractions import Fraction as F
from collections import Counter
import json,sys,math

def dot(a,b):return sum((x*y for x,y in zip(a,b)),F(0))
def transpose(a):return [list(r) for r in zip(*a)]
def mv(a,v):return [dot(r,v) for r in a]
def rank(a):
 a=[list(map(F,r)) for r in a];m=len(a);n=len(a[0]);row=0
 for col in range(n):
  pivot=next((i for i in range(row,m) if a[i][col]),None)
  if pivot is None:continue
  a[row],a[pivot]=a[pivot],a[row];q=a[row][col];a[row]=[x/q for x in a[row]]
  for i in range(m):
   if i!=row:
    q=a[i][col];a[i]=[x-q*y for x,y in zip(a[i],a[row])]
  row+=1
  if row==m:break
 return row

def solve(a,b):
 n=len(b);a=[list(map(F,r))+[F(v)] for r,v in zip(a,b)]
 for col in range(n):
  p=next(i for i in range(col,n) if a[i][col]);a[col],a[p]=a[p],a[col];v=a[col][col];a[col]=[x/v for x in a[col]]
  for i in range(n):
   if i!=col:
    v=a[i][col];a[i]=[x-v*y for x,y in zip(a[i],a[col])]
 return [r[-1] for r in a]

def grid(nx,nt):
 if not isinstance(nx,int) or not isinstance(nt,int) or nx<1 or nt<1:raise ValueError('Positive integer grid sizes required')
 nv=(nx+1)*(nt+1);node=lambda i,j:j*(nx+1)+i
 edges=[(node(i,j),node(i+1,j)) for j in range(nt+1) for i in range(nx)]+[(node(i,j),node(i,j+1)) for j in range(nt) for i in range(nx+1)]
 d=[]
 for tail,head in edges:
  r=[F(0)]*nv;r[tail]=-F(1);r[head]=F(1);d.append(r)
 h=lambda i,j:j*nx+i;v=lambda i,j:nx*(nt+1)+j*(nx+1)+i;c=[]
 for j in range(nt):
  for i in range(nx):
   r=[F(0)]*len(edges);r[h(i,j)]=1;r[v(i+1,j)]=1;r[h(i,j+1)]=-1;r[v(i,j)]=-1;c.append(r)
 return d,c

def project(d,w,error):
 if len(w)!=len(d) or any(x<=0 for x in w):raise ValueError('Positive link weights required')
 dp=[r[1:] for r in d];dt=transpose(dp)
 lap=[[sum((w[e]*dt[i][e]*dt[j][e] for e in range(len(w))),F(0)) for j in range(len(dt))] for i in range(len(dt))]
 chi=[F(0)]+solve(lap,[sum((w[e]*dt[i][e]*error[e] for e in range(len(w))),F(0)) for i in range(len(dt))]);grad=mv(d,chi)
 return chi,[x-y for x,y in zip(error,grad)]

def verify():
 out=Counter()
 for nx in [1,2,3]:
  for nt in [1,2,3]:
   d,c=grid(nx,nt);nv=(nx+1)*(nt+1);ne=len(d);nf=nx*nt
   assert all(dot(face,column)==0 for face in c for column in transpose(d));out['incidenceComplexes']+=1
   assert rank(d)==nv-1 and rank(c)==nf and ne-nf==nv-1;out['exactnessDimensionChecks']+=1
   w=[F(1+(i%5)) for i in range(ne)]
   for seed in [-2,0,1,3]:
    error=[F((i*i+seed*i+2)%11-5,3) for i in range(ne)]
    chi,r=project(d,w,error);wr=[x*y for x,y in zip(w,r)]
    assert mv(transpose(d),wr)==[0]*nv and mv(c,r)==mv(c,error);out['weightedNormalAndCurlChecks']+=1
    change=[F((i+seed)**2,7) for i in range(nv)];grad=mv(d,change)
    for lam in [-10,0,F(1,3),100]:
     shifted=[x+lam*g for x,g in zip(error,grad)];newchi,newr=project(d,w,shifted)
     assert newr==r;assert newchi==[x+lam*(v-change[0]) for x,v in zip(chi,change)];out['quotientOrbitChecks']+=1
    exact=mv(d,change);recovered,rr=project(d,w,exact);assert rr==[0]*ne;assert recovered==[v-change[0] for v in change];out['pureGaugeReconstructions']+=1
 d,c=grid(1,1)
 for weights in [[1,1,1,1],[1,2,4,8],[F(1,10),2,9,3]]:
  w=list(map(F,weights));face=c[0];den=sum((x*x/y for x,y in zip(face,w)),F(0))
  for seed in [-2,0,1,5]:
   error=[F(seed),F(2),F(-1,3),F(seed+1)];f=dot(face,error);chi,r=project(d,w,error)
   assert r==[x*f/(weight*den) for x,weight in zip(face,w)] and dot(r,[x*y for x,y in zip(w,r)])==f*f/den;out['singleFaceClosedForms']+=1
 # A removed face leaves a real extra loop, not a gradient.
 d,c=grid(3,3);ct=transpose(c);gram=[[dot(x,y) for y in c] for x in c];target=[F(i==4) for i in range(9)];theta=mv(ct,solve(gram,target))
 assert mv(c,theta)==target and mv(c[:4]+c[5:],theta)==[0]*8
 assert len(d)-rank(c[:4]+c[5:])==rank(d)+1
 assert any(project(d,[F(1)]*len(d),theta)[1]);out['puncturedDomainObstructions']+=1
 for nx in [3,4,5]:
  for nt in [1,2,3]:
   for kappa in [-2,F(1,3),1]:
    horizontal=[[kappa/nx]*nx for _ in range(nt+1)];vertical=[[F(0)]*nx for _ in range(nt)]
    for j in range(nt):
     for i in range(nx):assert horizontal[j][i]+vertical[j][(i+1)%nx]-horizontal[j+1][i]-vertical[j][i]==0
    assert all(sum(row)==kappa for row in horizontal)
    chi=[F(i*i-2*i) for i in range(nx)];assert sum(chi[(i+1)%nx]-chi[i] for i in range(nx))==0;out['periodicWrappingObstructions']+=1
 for n in [2,3,5]:
  chi=[F(i,n) for i in range(n+1)];path=[chi[i+1]-chi[i] for i in range(n)];assert sum(path)==1 and chi[0]==0 and chi[-1]!=0;out['restrictedBoundaryCounterexamples']+=1
 for nx,nt in [(0,1),(1,0),(-1,2),(F(1,2),1)]:
  try:grid(nx,nt)
  except ValueError:out['invalidGeometryRejections']+=1
  else:raise AssertionError('invalid grid accepted')
 for weights in [[1,1,0,1],[1,-1,1,1]]:
  try:project(*[grid(1,1)[0],weights,[1,2,3,4]])
  except ValueError:out['invalidWeightRejections']+=1
  else:raise AssertionError('invalid metric accepted')
 # Polynomial temporal amplitudes on a spatial sine eigenmode. Divide identities by length.
 for k in [1,2,3]:
  for a,b,c0 in [(1,0,0),(0,1,0),(0,0,1),(1,-2,3),(-2,3,1)]:
   for t in [F(0),F(1,4),F(1),F(3)]:
    amp=F(a)+b*t+c0*t*t;velocity=F(b)+2*c0*t;acceleration=F(2*c0)
    energy=(velocity*velocity+k*k*amp*amp)/4;source=acceleration+k*k*amp
    differentiated=velocity*acceleration/2+k*k*amp*velocity/2;work=velocity*source/2
    assert differentiated==work and work*work<=2*energy*source*source/2;out['forcedModeEnergyIdentities']+=1
 # 128-point real trigonometric quadrature is independent of the closed energy formula.
 for a in [-1,F(1,2),1]:
  for k in [1,2,3]:
   for t in [0,.25,.75,1.3]:
    n=128;length=2*math.pi;dz=length/n
    e=sum((float(a)*k*math.cos(k*(-math.pi+(i+.5)*dz))*math.cos(k*t))**2+(float(a)*k*math.sin(k*(-math.pi+(i+.5)*dz))*math.sin(k*t))**2 for i in range(n))*dz/2
    assert math.isclose(e,length*float(a)**2*k*k/4,rel_tol=2e-13);out['standingModeQuadratures']+=1
 for length in [F(1),F(2),F(3)]:
  for delta in [-F(1,10),F(0),F(1,10),F(2)]:
   for t in [F(0),F(1,4),F(1),F(3)]:
    gt=delta*t;energy=length*gt*gt/2;work=length*gt*delta
    derivative=length*delta*delta*t;assert work==derivative
    bound_squared=length*delta*delta*t*t/2;assert energy==bound_squared;out['uniformSourceHistoryEqualities']+=1
 for length in [F(1),F(2),F(3)]:
  for ratio in [F(0),F(1,4),F(1,2),F(1)]:
   t=ratio*length;energy=9*t**5/5;derivative=9*t**4
   # Integrate 9*(t-z)^4 coefficient-by-coefficient from z=0 to t.
   integral=sum(F(9*math.comb(4,j))*(-1)**j*t**(4-j)*t**(j+1)/F(j+1) for j in range(5))
   assert integral==energy and derivative==(-3*t*t)*(-3*t*t);out['boundaryEnergyInfluxIdentities']+=1
   for z in [F(0),t/2,t,length]:
    r=max(t-z,F(0));vtt=3*r*r;vzz=3*r*r;gtt=6*r;gzz=6*r;gt=3*r*r;gz=-3*r*r
    assert vzz-vtt==0 and gtt-gzz==0 and gz==-gt;out['incomingPotentialResidualChecks']+=1
 for gamma in [-2,0,F(1,3),3]:
  for t in [-1,0,1]:
   g=F(gamma);gt=gz=F(0);assert (gt*gt+gz*gz)/2==0 and g==gamma;out['constantMeanBlindCases']+=1
 return dict(sorted(out.items()))

def references():
 loops=[]
 for alpha in [-1,0,1]:
  for g2 in [-1,0,1]:
   for mode in ['original','coordinated','vector-only','scalar-only']:
    # b=1+t+t^2/2, g=1+t+g2*t^2/2; x[-.5,.5], y=z=1, t[0,.5].
    t=F(1,2);db=t+t*t/2;dg=t+g2*t*t/2;factor=int(mode in ['coordinated','scalar-only'])-int(mode in ['coordinated','vector-only'])
    loops.append(dict(alpha=alpha,g2=g2,mode=mode,baseCirculation=float(db/2),circulation=float(db/2+factor*alpha*dg)))
 standing=[dict(amplitude=float(a),k=k,tau=t,energy=2*math.pi*float(a)**2*k**4/4) for a in [-1,0,F(1,2)] for k in [1,2,3] for t in [0,.25,.75]]
 return dict(loops=loops,standing=standing)
if __name__=='__main__':
 result=references() if '--references' in sys.argv else verify()
 print(json.dumps(result,sort_keys=True))
