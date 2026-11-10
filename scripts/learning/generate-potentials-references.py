from pathlib import Path
import math,json,random
import numpy as np
from scipy.integrate import quad
rng=random.Random(53219);scalar=[];vector=[]
# Bounded electrostatic polynomial fields: V=.5 x.H.x+b.x+C.
# Numerical integration along straight and quadratic Bezier paths, independent
# finite differences, no product model import.
for i in range(36):
 raw=np.array([[rng.uniform(-2,2)for _ in range(3)]for _ in range(3)]);H=(raw+raw.T)/2
 b=np.array([rng.uniform(-1,1)for _ in range(3)]);C=rng.uniform(-3,3)
 p=np.array([rng.uniform(-1,1)for _ in range(3)]);q=np.array([rng.uniform(-1,1)for _ in range(3)]);control=np.array([rng.uniform(-2,2)for _ in range(3)])
 V=lambda x:.5*x@H@x+b@x+C;E=lambda x:-H@x-b
 paths=[(lambda t:p+t*(q-p),lambda t:q-p),(lambda t:(1-t)**2*p+2*t*(1-t)*control+t*t*q,lambda t:2*(1-t)*(control-p)+2*t*(q-control))]
 integrals=[quad(lambda t:float(E(at(t))@velocity(t)),0,1,epsabs=1e-11,epsrel=1e-11)[0]for at,velocity in paths]
 dv=float(V(q)-V(p));assert all(abs(v+dv)<1e-11 for v in integrals)
 scalar.append(dict(matrix=H.tolist(),linear=b.tolist(),referenceConstant=C,start=p.tolist(),end=q.tolist(),bezierControl=control.tolist(),potentialDifference=dv,startPotential=float(V(p)),endPotential=float(V(q)),endField=E(q).tolist(),divergence=float(-np.trace(H)),fieldIntegrals=integrals,maxDifference=max(abs(v+dv)for v in integrals)))
# Uniform-current-cylinder Az, normalized with mu=1 and Az(a)=0.
# Curl of Az plus gradient of a smooth single-valued chi is obtained by FD.
for i in range(36):
 a=[.5,1.,2.][i%3];I=[-2.,0.,1.][i//3%3]
 while True:
  p=np.array([rng.uniform(-3,3)for _ in range(3)]);r=float(np.hypot(*p[:2]))
  if abs(r-a)>.1:break
 alpha=rng.uniform(-1,1);g=rng.uniform(-1,1)
 def Az(r):return I/(4*math.pi)*(1-(r/a)**2)if r<=a else -I/(2*math.pi)*math.log(r/a)
 def A(x):
  u,v,w=x;return np.array([alpha*u+g*v*w,alpha*v+g*u*w,Az(float(np.hypot(u,v)))+alpha*w+g*u*v])
 h=1e-4;derivatives=[]
 for j in range(3):
  axis=np.eye(3)[j]*h;derivatives.append((-A(p+2*axis)+8*A(p+axis)-8*A(p-axis)+A(p-2*axis))/(12*h))
 dx,dy,dz=derivatives;curl=np.array([dy[2]-dz[1],dz[0]-dx[2],dx[1]-dy[0]])
 coefficient=I/(2*math.pi*(a*a if r<=a else r*r));expected=coefficient*np.array([-p[1],p[0],0])
 error=float(np.max(np.abs(curl-expected)));assert error<1e-9
 # No infinity-zero Az convention: exterior log grows without bound.
 vector.append(dict(radius=a,current=I,observation=p.tolist(),gaugeQuadratic=alpha,gaugeCubic=g,vectorPotential=A(p).tolist(),independentCurl=curl.tolist(),ampereField=expected.tolist(),difference=error))
rectangles=[]
for i in range(24):
 a=[.5,1.,2.][i%3];I=[-2.,0.,1.][i//3%3];y=[0.,.2,.7][i%3]
 left=rng.uniform(-2,0);right=rng.uniform(.25,3);bottom=rng.uniform(-2,0);top=rng.uniform(.25,2)
 def az(x):
  r=math.hypot(x,y)
  return I/(4*math.pi)*(1-(r/a)**2)if r<=a else -I/(2*math.pi)*math.log(r/a)
 def by(x):
  r=math.hypot(x,y)
  return I*x/(2*math.pi*(a*a if r<=a else r*r))
 points=[v for v in [-math.sqrt(a*a-y*y),math.sqrt(a*a-y*y)]if left<v<right]if abs(y)<a else []
 flux=-(top-bottom)*quad(by,left,right,points=points,epsabs=1e-12,epsrel=1e-12)[0]
 circulation=(top-bottom)*(az(right)-az(left));assert abs(flux-circulation)<1e-11
 rectangles.append(dict(radius=a,current=I,y=y,left=left,right=right,bottom=bottom,top=top,independentMagneticFluxOverMu=flux,sourceCirculationOverMu=circulation))
result=dict(provenance='Independent NumPy polynomial fields, SciPy path quadrature and five-point curl of cylinder vector potential; no product imports',scalarPaths=scalar,vectorCurls=vector,rectangles=rectangles,maxWorkDifference=max(x['maxDifference']for x in scalar),maxCurlDifference=max(x['difference']for x in vector),limits=['Independent references for the static bounded-domain teaching models; no product imports.','Static single-valued smooth gauge; time-dependent SI transformation must be derived separately.','Curl-free is locally sufficient; global scalar potential needs topology assumptions.','Cylinder vector potential uses Az(a)=0,not an impossible infinity-zero logarithmic reference.'])
Path('scripts/fixtures/potentials-independent.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items()if k.startswith('max')}|{'scalarPaths':len(scalar),'vectorCurls':len(vector),'rectangles':len(rectangles)}))
