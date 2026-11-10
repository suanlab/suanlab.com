"""Independent exact Fraction references for binary products and kernel moments.
Particle positions are the declared ordinary binary64 geometry; all subsequent
kernel and moment arithmetic is exact rational arithmetic before projection.
"""
from fractions import Fraction as F
from pathlib import Path
import json,math,argparse,random
p=argparse.ArgumentParser();p.add_argument('--output',default='scripts/fixtures/continuum-binary-audit-independent.json');target=Path(p.parse_args().output)
if target.exists():raise SystemExit('Refusing to overwrite reference')
def f(x):return F.from_float(float(x))
def product(xs):
 v=F(1)
 for x in xs:v*=f(x)
 return v
def obs(q):
 if not q:return dict(value=0.,sign=0,logMagnitude=None)
 try:value=float(q)
 except OverflowError:value=None
 if value==0 or value is not None and not math.isfinite(value):value=None
 return dict(value=value,sign=1 if q>0 else -1,logMagnitude=math.log(abs(q.numerator))-math.log(q.denominator))
tiny=float.fromhex('0x0.0000000000001p-1022');cases=[]
inputs=[([[tiny,.5]],[[1.]]),([[tiny,.5,4.]],[[1.]]),([[tiny,tiny],[-tiny,tiny],[tiny,tiny,tiny]],[[1.]]),([[1e308,1e308,1e-308]],[[1.]]),([[tiny]],[[tiny]]),([[tiny,tiny]],[[tiny]]),([[3.,tiny]],[[2.]]),([[5.,tiny]],[[2.]]),([[-3.,tiny]],[[2.]]),([[1.],[-math.nextafter(1.,2.)]],[[1.]]),([[1.,tiny],[-1.,tiny]],[[1.]]),([[math.pi,math.pi]],[[3.]])]
rng=random.Random(261112)
for i in range(28):
 scale=[1.,1e-200,tiny,1e200][i%4];inputs.append(([[scale,rng.uniform(-2,2)],[scale,rng.uniform(-2,2)],[scale,rng.uniform(-2,2)]],[[rng.uniform(.1,4)]]))
for terms,den in inputs:cases.append(dict(terms=terms,denominator=den,expected=obs(sum(map(product,terms))/sum(map(product,den)))))
def kernel(x,h):
 x=f(x);h=f(h)
 if abs(x)>=h:return dict(value=F(0),spatialDerivative=F(0),bandwidthDerivative=F(0))
 q=x/h
 return dict(value=F(15,16)/h*(1-q*q)**2,spatialDerivative=-F(15,4)*q*(1-q*q)/h**2,bandwidthDerivative=-F(15,16)*(1-q*q)*(1-5*q*q)/h**2)
kernels=[]
for x,h in [(0.,1.),(tiny,2.),(tiny,1.),(tiny,.02),(math.nextafter(1.,0.),1.),(1.,1.),(-1.,1.),(1/math.sqrt(5),1.),(.2,.7),(-.2,.7),(16.,.02),(.02,.02)]:kernels.append(dict(x=x,h=h,expected={k:obs(v)for k,v in kernel(x,h).items()}))
particles=[]
base=[dict(mass=1.,position=0.,velocity=1.),dict(mass=1.,position=0.,velocity=-1.)]
params=[(base,0.,0.,1.,0.),([{**base[0],'mass':3.},base[1]],0.,0.,1.,0.),([{**base[0],'velocity':tiny},{**base[1],'velocity':-tiny}],0.,0.,1.,0.),([{**base[0],'velocity':tiny},{**base[1],'velocity':tiny}],0.,0.,1.,0.),([{**base[0],'velocity':1e-200},{**base[1],'velocity':-1e-200}],0.,0.,1.,0.),([dict(mass=4.,position=0.,velocity=tiny)],tiny,0.,2.,0.),([dict(mass=1.,position=0.,velocity=0.)],0.,0.,2.,tiny),([dict(mass=1.,position=0.,velocity=tiny)],0.,0.,1.,tiny),([dict(mass=1.,position=0.,velocity=1e-200),dict(mass=1.,position=0.,velocity=math.nextafter(1e-200,2.))],0.,0.,1.,0.),([base[0],{**base[1],'mass':math.nextafter(1.,2.)}],0.,0.,1.,0.),(base,1/math.sqrt(5),0.,1.,tiny),(base,8.,0.,.02,0.)]
for i in range(12):params.append((base,rng.uniform(-.7,.7),rng.uniform(-.2,.2),rng.uniform(.4,2),rng.uniform(-.5,.5)))
for ps,x,time,h,rate in params:
 contributions=[]
 for p in ps:
  position=p['position']+p['velocity']*time;dx=x-position;k=kernel(dx,h);contributions.append((f(p['mass']),f(p['velocity']),k))
 rho=sum(m*k['value']for m,v,k in contributions);j=sum(m*v*k['value']for m,v,k in contributions);Q=sum(m*v*v*k['value']for m,v,k in contributions)
 u=j/rho if rho else None;S=sum(m*(v-u)**2*k['value']for m,v,k in contributions)if rho else None
 rhox=sum(m*k['spatialDerivative']for m,v,k in contributions);jx=sum(m*v*k['spatialDerivative']for m,v,k in contributions);Qx=sum(m*v*v*k['spatialDerivative']for m,v,k in contributions);width=f(rate)*sum(m*k['bandwidthDerivative']for m,v,k in contributions);momentWidth=f(rate)*sum(m*v*k['bandwidthDerivative']for m,v,k in contributions)
 values=dict(density=rho,flux=j,velocity=u,momentumFlux=Q,meanMomentumTransport=rho*u*u if rho else None,dispersion=S,densityX=rhox,densityT=-jx+width,fluxX=jx,smoothingRate=width,continuityResidual=width,momentumT=-Qx+momentWidth,momentumFluxX=Qx,momentumSmoothingRate=momentWidth,momentumResidual=momentWidth)
 particles.append(dict(particles=ps,x=x,time=time,h=h,bandwidthRate=rate,expected={k:None if v is None else obs(v)for k,v in values.items()}))
target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(dict(method=__doc__,binaryCases=cases,kernelCases=kernels,particleCases=particles),indent=2)+'\n');print('Wrote40 binary,12 kernel,24 particle exact rational cases.')
