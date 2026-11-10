"""Independent references for prescribed static spherical/cylindrical sources.
Coulomb shell-potential quadrature plus independent finite differences; Biot-Savart
filament geometry,not a product implementation import or an Ampere-field integrand.
"""
from pathlib import Path
import json,math,itertools,random
from scipy.integrate import quad
spheres=[];cylinders=[]
for total,radius,fraction in itertools.product([-2.,0.,1.],[.5,1.,2.],[0.,.2,.8,1.,1.2,2.,4.]):
 r=radius*fraction
 # Angular Coulomb average over a spherical shell gives1/max(r,s).
 # Integrate the shells numerically,then differentiate the potential independently.
 def potential(x):
  x=abs(x)
  return quad(lambda s:0. if s==0 else 3*total/radius**3*s*s/max(x,s),0,radius,points=[x]if 0<x<radius else None,epsabs=1e-12,epsrel=2e-12)[0]
 h=radius*1e-5
 negative_gradient=-(-potential(r+2*h)+8*potential(r+h)-8*potential(r-h)+potential(r-2*h))/(12*h)
 expected=total*r/radius**3 if r<=radius else total/r**2
 tolerance=(2e-5 if fraction==1 else 2e-9)*max(1,abs(expected));assert abs(negative_gradient-expected)<tolerance
 spheres.append(dict(totalCharge=total,radius=radius,observationRadius=r,potentialOverK=potential(r),negativePotentialGradientOverK=negative_gradient,gaussPredictionOverK=expected,boundary=fraction==1,difference=abs(negative_gradient-expected)))
for current,radius,fraction in itertools.product([-2.,0.,1.],[.5,1.,2.],[0.,.2,.8,1.,1.2,2.,4.]):
 r=radius*fraction;density=current/(math.pi*radius**2)
 # Independent Biot-Savart superposition of infinitely long filaments.
 # Polar coordinates centered on the observation remove the integrable1/s
 # singularity via the area Jacobian; ray/disk intersection lengths remain.
 if r<=radius:
  def fun(theta):
   co=math.cos(theta);length=-r*co+math.sqrt(max(0.,radius**2-r**2*math.sin(theta)**2))
   return -co*length
  angular=quad(fun,0,2*math.pi,points=[math.pi/2,3*math.pi/2],epsabs=1e-12,epsrel=2e-12)[0]
 else:
  angle=math.asin(radius/r)
  angular=quad(lambda psi:math.cos(psi)*2*math.sqrt(max(0.,radius**2-r**2*math.sin(psi)**2)),-angle,angle,epsabs=1e-12,epsrel=2e-12)[0]
 field=density*angular/(2*math.pi);expected=current*r/(2*math.pi*radius**2)if r<=radius else current/(2*math.pi*r)
 assert abs(field-expected)<2e-11*max(1,abs(expected))
 cylinders.append(dict(totalCurrent=current,radius=radius,observationRadius=r,biotSavartFieldOverMu0=field,amperePredictionOverMu0=expected,difference=abs(field-expected)))

rng=random.Random(74193);vectors=[]
def sphere_potential(q,a,p):
 r=math.sqrt(sum(x*x for x in p))
 return quad(lambda u:0. if u==0 else 3*q/a**3*u*u/max(r,u),0,a,points=[r]if 0<r<a else None,epsabs=1e-12,epsrel=2e-12)[0]
for i in range(24):
 a=[.5,1.,2.][i%3];q=[-2.,0.,1.][(i//3)%3]
 while True:
  position=[rng.uniform(-2,2)for _ in range(3)]
  if abs(math.sqrt(sum(x*x for x in position))-a)>.1:break
 background=[rng.uniform(-.5,.5)for _ in range(3)];h=2e-5*a
 def total_potential(p):return sphere_potential(q,a,p)-sum(x*y for x,y in zip(background,p))
 gradient=[]
 for j in range(3):
  def at(step):
   p=position.copy();p[j]+=step*h;return total_potential(p)
  gradient.append(-(-at(2)+8*at(1)-8*at(-1)+at(-2))/(12*h))
 vectors.append(dict(totalCharge=q,radius=a,observation=position,backgroundField=background,totalPotential=total_potential(position),negativeIndependentGradient=gradient))
convergence=[]
for h in[.01,.005,.0025,.00125]:
 fun=lambda r:sphere_potential(1.,1.,[r,0,0])
 gradient=-(-fun(1+2*h)+8*fun(1+h)-8*fun(1-h)+fun(1-2*h))/(12*h)
 convergence.append(dict(step=h,negativePotentialGradient=gradient,exactBoundaryFieldOverK=1.,difference=abs(gradient-1.)))
assert all(convergence[i+1]['difference']<convergence[i]['difference']for i in range(3))
result=dict(provenance='Independent SciPy Coulomb shell quadrature and five-point potential gradients; observer-centered Biot-Savart filament integration; no product code imports',spheres=spheres,cylinders=cylinders,vectorSpheres=vectors,boundaryConvergence=convergence,limits=['Uniform prescribed nonconducting sphere,not conductor equilibrium.','Infinite uniform steady current cylinder,not a finite open wire or a constitutive model.','k and mu0 are factored out.','Finite differences across the interface have reduced order; their observed differences are not certified bounds.'])
Path('scripts/fixtures/electro-magnetostatics-independent.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(spheres=len(spheres),cylinders=len(cylinders),vectorSpheres=len(vectors),boundaryConvergence=len(convergence),maxSphereAwayFromBoundaryDifference=max(x['difference']for x in spheres if not x['boundary']),maxSphereBoundaryDifference=max(x['difference']for x in spheres if x['boundary']),maxBiotSavartDifference=max(x['difference']for x in cylinders))))
