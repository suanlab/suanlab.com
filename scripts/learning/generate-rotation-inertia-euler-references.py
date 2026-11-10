from pathlib import Path
from fractions import Fraction as Q
from scipy.integrate import solve_ivp
import numpy as np,json,random,math
rng=random.Random(20261104417);refs=[]
special=[[1,1,1,10,10,10,0,100],[1,1,2,10,5,10,6,100],[3,2,1,0,10,0,0,100],[3,2,1,1,10,0,0,100],[1,5,1,-20,20,20,-12,100],[5,5,5,0,0,0,6,100],[1,1,1,0,0,10,6,100],[5,1,4,20,-20,20,-6,0]]
for i in range(180):
 p=special[i]if i<len(special)else[rng.randint(1,5),rng.randint(1,5),rng.randint(1,5),rng.randint(-20,20),rng.randint(-20,20),rng.randint(-20,20),rng.randint(-12,12),rng.randint(0,100)]
 A,B,C,wx,wy,wz,ac,tc=p;I=np.array([B+C,C+A,A+B],float);w0=np.array([wx,wy,wz],float)/10;T=tc/10;theta=ac*math.pi/12;ct,st=math.cos(theta),math.sin(theta);R0=np.array([[ct,-st,0],[st,ct,0],[0,0,1.]])
 def rhs(t,z):
  w=z[:3];R=z[3:].reshape(3,3);x,y,v=w;skew=np.array([[0,-v,y],[v,0,-x],[-y,x,0.]])
  # Body momentum transport rather than explicit component Euler formulas.
  wd=-np.cross(w,I*w)/I
  return np.concatenate([wd,(R@skew).flatten()])
 z0=np.concatenate([w0,R0.flatten()])
 if T:
  sol=solve_ivp(rhs,(0,T),z0,method='DOP853',rtol=2e-13,atol=3e-15);assert sol.success;z=sol.y[:,-1]
 else:z=z0
 w=z[:3];rot=z[3:].reshape(3,3);energy=sum(Q(int(v))*Q(k,10)**2/2 for v,k in zip(I,[wx,wy,wz]));l2=sum(Q(int(v))**2*Q(k,10)**2 for v,k in zip(I,[wx,wy,wz]));refs.append(dict(values=p,energy=str(energy),angularNormSquared=str(l2),bodyOmega=w.tolist(),rotation=rot.tolist(),labAngularMomentum=(rot@(I*w)).tolist()))
Path('scripts/fixtures/rotation-inertia-euler-independent.json').write_text(json.dumps(refs,separators=(',',':'))+'\n');print('180 DOP853 body-transport + rotation-matrix references, independent of quaternion/RK4 implementation.')
