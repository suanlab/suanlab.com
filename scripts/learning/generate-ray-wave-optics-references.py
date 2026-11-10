"""Portable product-free references: sequential rays and direct 2D Fresnel quadrature.
Requires numpy, scipy and sympy. Run from repository root with python3.
Fields have exp(+ikz-iwt) carrier removed; input amplitude is relative, coordinates m.
"""
from pathlib import Path
import itertools,json
import numpy as np
from scipy.integrate import quad
import sympy as sp

def packed(z):return [float(z.real),float(z.imag)]
def cq(fun):
 return quad(lambda t:float(fun(t).real),-10,10,epsabs=2e-12,epsrel=2e-12,limit=600)[0]+1j*quad(lambda t:float(fun(t).imag),-10,10,epsabs=2e-12,epsrel=2e-12,limit=600)[0]
rays=[]
for u,z,f,x,central,half in itertools.product([.02,.08,.2],[0,.025,.15],[-.04,.02,.08,None],[-.0007,.001],[0,.013],[0,.01]):
 points=[]
 for theta in np.linspace(central-half,central+half,9):
  at=x+u*theta;out=theta-(0 if f is None else at/f)
  points.append(dict(inputAngle=float(theta),lensHeight=float(at),outputAngle=float(out),screenHeight=float(at+z*out)))
 v=None if f is None or u==f else f*u/(u-f)
 rays.append(dict(system=dict(objectDistance=u,screenDistance=z,focalLength=f,objectHeight=x,centralAngle=central,fanHalfAngle=half),rays=points,imageDistance=v,imageHeight=None if v is None else -v*x/u))
beams=[];max_quadrature_error=0
for index,(lam,w0,f,u,z) in enumerate(itertools.product([500e-9,800e-9],[40e-6,120e-6],[-.04,.02,.08,None],[0,.03],[.01,.025,.08])):
 amp=[0,.7,1.8][index%3];k=2*np.pi/lam
 du=1+2j*u/(k*w0*w0);au=1/(w0*w0*du);wlens=1/np.sqrt(au.real);alens=au+(0 if f is None else 1j*k/(2*f))
 dz=1+2j*z*alens/k;afinal=alens/dz;prefix=1/np.sqrt(du);pout=amp/(du*dz);width=1/np.sqrt(afinal.real)
 # Actual two transverse integrals. The finite ±10 input radii omit <exp(-100) tails.
 def numerical1d(x):return prefix*wlens/np.sqrt(1j*lam*z)*cq(lambda t:np.exp(-au*(wlens*t)**2-(0 if f is None else 1j*k*(wlens*t)**2/(2*f))+1j*k*(x-wlens*t)**2/(2*z)))
 atzero=numerical1d(0);observations=[]
 for ratio in [0,.5,1.5]:
  x=ratio*width;field=amp*numerical1d(x)*atzero;closed=pout*np.exp(-afinal*x*x)
  max_quadrature_error=max(max_quadrature_error,float(abs(field-closed)))
  observations.append(dict(radius=float(x),field=packed(field),intensity=float(abs(field)**2)))
 radial=quad(lambda t:2*np.pi*width**2*t*abs(pout*np.exp(-afinal*(width*t)**2))**2,0,10,epsabs=1e-18,epsrel=2e-12)[0]
 # Fit width²(z)=c0+c1*z+c2*z² using coefficient propagation at three independent locations.
 samples=[]
 for zz in [-.02,0,.02]:samples.append(1/(alens/(1+2j*zz*alens/k)).real)
 c0=samples[1];c1=(samples[2]-samples[0])/.04;c2=(samples[2]+samples[0]-2*c0)/(.0008)
 waist_position=-c1/(2*c2);waist_radius=np.sqrt(c0-c1*c1/(4*c2))
 beams.append(dict(system=dict(wavelength=lam,waistRadius=w0,amplitude=amp,preDistance=u,postDistance=z,focalLength=f),width=float(width),coefficient=packed(afinal),prefactor=packed(pout),observations=observations,intensityIntegral=float(radial),waistCoordinate=float(waist_position),waistRadius=float(waist_radius)))
u,v,f,x0,s,x=sp.symbols('u v f x0 s x',nonzero=True,real=True)
phase=(s-x0)**2/(2*u)-s*s/(2*f)+(x-s)**2/(2*v)
image=f*u/(u-f);stationary=sp.factor(sp.diff(phase,s).subs({v:image,x:-image*x0/u}));assert stationary==0
assert max_quadrature_error<2e-9
report=dict(provenance=dict(generator='scripts/learning/generate-ray-wave-optics-references.py',productImports=False,method='Sequential lens geometry; two actual 1D Fresnel quadratures form 2D field; radial power quadrature; width-square polynomial fit; SymPy stationary phase',convention='exp(+ikz-iwt); relative-field units; SI lengths',symbolicImagingDerivative=str(stationary),maxQuadratureClosedFieldDifference=max_quadrature_error),rays=rays,beams=beams)
path=Path('scripts/fixtures/ray-wave-optics-independent.json');path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(raySystems=len(rays),gaussianSystems=len(beams),complexObservations=3*len(beams),maxQuadratureClosedFieldDifference=max_quadrature_error,symbolicImagingDerivative=str(stationary)),indent=2))
