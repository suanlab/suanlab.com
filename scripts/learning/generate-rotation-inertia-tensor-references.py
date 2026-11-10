from pathlib import Path
from fractions import Fraction as Q
import json,random,math
import numpy as np
rng=random.Random(20261104416)
rows=[]
special=[[1,1,1,2,2,2,0,10,10,0,0,0],[1,1,1,2,4,6,0,10,10,0,0,0],[1,1,1,2,4,6,0,10,10,0,10,0],[1,1,1,2,4,6,1,10,10,0,0,6],[4,4,4,1,1,1,-4,0,0,0,-20,-12],[1,1,1,2,2,2,0,10,0,0,10,0]]
def cross(a,b):return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def add(a,b):return[x+y for x,y in zip(a,b)]
def sub(a,b):return[x-y for x,y in zip(a,b)]
def dot(a,b):return sum((x*y for x,y in zip(a,b)),Q(0))
for i in range(600):
 p=special[i]if i<len(special)else[rng.randint(1,4),rng.randint(1,4),rng.randint(1,4),rng.randint(1,6),rng.randint(1,6),rng.randint(1,6),rng.randint(-4,4),rng.randint(-20,20),rng.randint(-20,20),rng.randint(-20,20),rng.randint(-20,20),rng.randint(-12,12)]
 m1,m2,m3,ac,bc,cc,sc,wxc,wyc,wzc,dc,oc=p;a,b,c,sh=[Q(x,2)for x in [ac,bc,cc,sc]];omega=[Q(x,10)for x in [wxc,wyc,wzc]];O=[Q(dc,10),Q(0),Q(0)];points=[(m1,[a,0,0]),(m1,[-a,0,0]),(m2,[0,b,0]),(m2,[0,-b,0]),(m3,[sh,0,c]),(m3,[-sh,0,-c])];M=2*(m1+m2+m3)
 IC=[[Q(0)for _ in range(3)]for _ in range(3)];IO=[[Q(0)for _ in range(3)]for _ in range(3)];P=[Q(0)]*3;LO=[Q(0)]*3;LC=[Q(0)]*3;KC=Q(0);KO=Q(0)
 for m,r in points:
  ro=sub(r,O);v=cross(omega,ro);mom=[m*x for x in v];P=add(P,mom);LO=add(LO,cross(ro,mom));LC=add(LC,cross(r,mom));KO+=Q(m,2)*dot(v,v);vcom=cross(omega,r);KC+=Q(m,2)*dot(vcom,vcom)
  for j in range(3):
   for k in range(3):
    # Diagonal entries from squared distances to axes, no matrix formula.
    IC[j][k]+=m*(sum(r[l]**2 for l in range(3)if l!=j)if j==k else-r[j]*r[k]);IO[j][k]+=m*(sum(ro[l]**2 for l in range(3)if l!=j)if j==k else-ro[j]*ro[k])
 vc=[x/M for x in P];ktrans=Q(M,2)*dot(vc,vc);axis=None if dot(omega,omega)==0 else 2*KO/dot(omega,omega)
 theta=oc*math.pi/12;ct=math.cos(theta);st=math.sin(theta);rot=np.array([[ct,-st,0],[st,ct,0],[0,0,1.]])
 IOn=np.array(IO,dtype=float);eigen=np.linalg.eigvalsh(IOn).tolist();lab=rot@IOn@rot.T;labomega=rot@np.array(omega,dtype=float);labL=rot@np.array(LO,dtype=float)
 # Evaluate directional inertia from geometric distances to each axis.
 directional=[]
 for k in [0,15,30,45,60,75,90,120,150,180]:
  alpha=k*math.pi/180;n=np.array([math.cos(alpha),0,math.sin(alpha)])
  value=sum(m*(np.dot(np.array(sub(r,O),dtype=float),np.array(sub(r,O),dtype=float))-np.dot(np.array(sub(r,O),dtype=float),n)**2)for m,r in points)
  directional.append([k,float(value)])
 rows.append(dict(values=p,exact=dict(mass=str(M),tensorCOM=[[str(x)for x in row]for row in IC],tensorPivot=[[str(x)for x in row]for row in IO],angularMomentumCOM=[str(x)for x in LC],angularMomentumPivot=[str(x)for x in LO],linearMomentum=[str(x)for x in P],centerVelocity=[str(x)for x in vc],kineticCOM=str(KC),kineticPivot=str(KO),translationKinetic=str(ktrans),axisInertia=None if axis is None else str(axis)),eigenvalues=eigen,labTensor=lab.tolist(),labOmega=labomega.tolist(),labAngularMomentum=labL.tolist(),directional=directional))
Path('scripts/fixtures/rotation-inertia-tensor-independent.json').write_text(json.dumps(rows,separators=(',',':'))+'\n');print('600 independent Fraction point-velocity/angular/energy and NumPy spectral/coordinate/directional references generated. Unregistered draft.')
