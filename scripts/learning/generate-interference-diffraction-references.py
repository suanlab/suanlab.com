"""Portable independent references; run from repository root. No product imports.
SciPy QUADPACK handles actual field averages/aperture integrals; Fresnel special
functions independently check the unfactored propagation kernel.
"""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import quad
from scipy.special import fresnel
import sympy as sp
import mpmath as mp
rng=np.random.default_rng(20261107)
def cq(f,a,b):return quad(lambda x:float(f(x).real),a,b,epsabs=2e-12,epsrel=2e-12)[0]+1j*quad(lambda x:float(f(x).imag),a,b,epsabs=2e-12,epsrel=2e-12)[0]
def pair(z):return [float(z.real),float(z.imag)]
interference=[]
for case in range(48):
 a,b=rng.uniform(.1,2,2);omega1,omega2=rng.uniform(0,16,2);T=[0.,.2,1.,4.,8.][case%5];center=float(rng.uniform(-2,2));phase=float(rng.uniform(-np.pi,np.pi));population=[{'phase':float(rng.uniform(-np.pi,np.pi)),'weight':float(rng.integers(1,5))} for _ in range(1+case%3)]
 if case%7==0:omega2=omega1
 if case%11==0:a=0.
 if case==47:a=b=0.
 system={'amplitude1':float(a),'amplitude2':float(b),'omega1':float(omega1),'omega2':float(omega2),'windowDuration':T,'timeCenter':center,'phaseDifference':phase,'population':population}
 total=sum(p['weight'] for p in population)
 def average(offset,real=False):
  out=0.
  for p in population:
   field=lambda t:a*np.exp(-1j*omega1*t)+b*np.exp(1j*(phase+p['phase']+offset-omega2*t))
   f=(lambda t:field(t).real**2) if real else (lambda t:abs(field(t))**2)
   out+=p['weight']/total*(f(center) if T==0 else quad(f,center-T/2,center+T/2,epsabs=2e-12,epsrel=2e-12,limit=300)[0]/T)
  return float(out)
 i0,ip,i90=average(0),average(np.pi),average(np.pi/2);base=(i0+ip)/2;mod=np.hypot((i0-ip)/2,i90-base)
 gamma=sum(p['weight']/total*np.exp(1j*p['phase']) for p in population)
 interference.append({'system':system,'meanIntensity':i0,'realSquareMean':average(0,True),'baseline':float(a*a+b*b),'maximum':float(base+mod),'minimum':float(base-mod),'visibility':None if base==0 else float(mod/base),'phaseMoment':pair(gamma)})
apertures=[];max_difference=0.
for case in range(36):
 slits=1+case%2;F=[.01,.1,.5,1.,5.][case%5];width=[20e-6,50e-6,200e-6][case%3];wavelength=[400e-9,500e-9,800e-9][(case//3)%3];sep=[1.,2.,4.][(case//2)%3];a=[0.,.5,1.,2.][(case//3)%4];b=[0.,.25,1.,1.5][case%4];phase=[0.,np.pi/2,np.pi][(case//4)%3]
 system={'slits':slits,'fresnelNumber':F,'slitWidth':width,'wavelength':wavelength,'separationRatio':sep,'amplitude1':a,'amplitude2':b,'relativePhase':phase}
 sources=[(0.,a,0.)] if slits==1 else [(-sep/2,a,0.),(sep/2,b,phase)]
 states=[]
 for eta in [-2.,-1.75,-1.5,-1.,-.5,-.125,0.,.125,.5,1.,1.5,1.75,2.]:
  X=eta/F;near=far=special=0j;pref=np.exp(-1j*np.pi/4)*np.sqrt(F)
  for center,amplitude,phi in sources:
   lo,hi=center-.5,center+.5
   near+=pref*amplitude*cq(lambda u:np.exp(1j*(np.pi*F*(X-u)**2+phi)),lo,hi)
   far+=pref*np.exp(1j*np.pi*eta*eta/F)*amplitude*cq(lambda u:np.exp(1j*(phi-2*np.pi*eta*u)),lo,hi)
   slo,clo=fresnel(np.sqrt(2*F)*(lo-X));shi,chi=fresnel(np.sqrt(2*F)*(hi-X))
   special+=amplitude*np.exp(1j*(phi-np.pi/4))/np.sqrt(2)*(chi-clo+1j*(shi-slo))
  delta=float(abs(near-special));max_difference=max(max_difference,delta);assert delta<2e-11
  states.append({'eta':eta,'fresnel':pair(near),'fraunhofer':pair(far),'fresnelIntensity':float(abs(near)**2),'fraunhoferIntensity':float(abs(far)**2),'directSpecialDifference':delta})
 apertures.append({'system':system,'states':states,'expectedTransmittedNorm':width*sum(amplitude**2 for _,amplitude,_ in sources)})
phi=sp.symbols('phi',real=True);root=sp.sqrt(3);i1=(2+root)/2;i2=(2-root)/2
assert sp.simplify(i1+i2+2*sp.sqrt(i1*i2)*sp.cos(phi)-(2+sp.cos(phi)))==0
mp.mp.dps=70
tiny_system={'amplitude1':1.,'amplitude2':.7,'omega1':2.,'omega2':3.,'phaseDifference':.3,'timeCenter':16.,'windowDuration':1e-18,'population':[{'phase':0.,'weight':1.}]}
def tiny_field(u):
 t=mp.mpf(16)+mp.mpf('1e-18')*u
 return mp.exp(-2j*t)+mp.mpf('.7')*mp.exp(1j*(mp.mpf('.3')-3*t))
tiny_reference={'system':tiny_system,'realSquareMean':float(mp.quad(lambda u:mp.re(tiny_field(u))**2,[-mp.mpf('.5'),mp.mpf('.5')])),'meanIntensity':float(mp.quad(lambda u:abs(tiny_field(u))**2,[-mp.mpf('.5'),mp.mpf('.5')])),'decimalPrecision':70}
out={'provenance':{'generator':'scripts/learning/generate-interference-diffraction-references.py','independence':'No product imports. Direct QUADPACK time averages,visibility inferred from three independently averaged phase offsets,unfactored Fresnel kernel quadrature plus scipy.special.fresnel,far-field aperture quadrature.','interferenceCases':48,'apertureSystems':36,'aperturePoints':468,'maxDirectSpecialDifference':max_difference,'symbolicNonidentifiability':True,'tinyExposureDecimalPrecision':70},'interference':interference,'apertures':apertures,'tinyExposure':tiny_reference}
Path('scripts/fixtures/interference-diffraction-independent.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['provenance'],indent=2))
