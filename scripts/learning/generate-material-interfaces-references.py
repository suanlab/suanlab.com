"""Independent vector integrals, boundary solves, spacetime differences and180-digit references.
Dependencies: Python NumPy, SciPy, mpmath. No product imports.
Run: python3 scripts/learning/generate-material-interfaces-references.py
"""
from pathlib import Path
import math,json,itertools
import numpy as np
from scipy.integrate import quad
import mpmath as mp
mp.mp.dps=180
mu0=4*math.pi*1e-7;eps0=1/(mu0*299792458**2)
static=[];waves=[]
for e1,e2,D1,free,Et in itertools.product([1.,2.,4.],[1.,2.,4.],[-2e-9,0.,2e-9],[-1e-9,0.,1e-9],[-1.,0.,1.]):
 eps1,eps2=eps0*e1,eps0*e2;E1=np.array([Et,0.,D1/eps1]);E2=np.array([Et,0.,(D1+free)/eps2]);P1=(eps1-eps0)*E1;P2=(eps2-eps0)*E2;bound=float((P1-P2)[2]);R=.7;height=.4;span=.6
 E=lambda z:E1 if z<0 else E2
 D=lambda z:(eps1 if z<0 else eps2)*E(z)
 def surface_flux(F):
  top=quad(lambda r:float(F(height)[2])*2*math.pi*r,0,R,epsabs=1e-16)[0]
  bottom=-quad(lambda r:float(F(-height)[2])*2*math.pi*r,0,R,epsabs=1e-16)[0]
  sides=quad(lambda theta:sum(float(F(z)@np.array([math.cos(theta),math.sin(theta),0.]))*height*R for z in[-height/2,height/2]),0,2*math.pi,epsabs=1e-16)[0]
  return top+bottom+sides
 dflux=surface_flux(D);eflux=surface_flux(lambda z:eps0*E(z));area=math.pi*R*R
 # Oriented x-z rectangle: lower edge +x, upper edge -x, normal -y.
 bottom=quad(lambda x:float(E(-height)[0]),-span/2,span/2)[0];top=-quad(lambda x:float(E(height)[0]),-span/2,span/2)[0]
 right=quad(lambda z:float(E(z)[2]),-height,height,points=[0])[0];left=-right;line=bottom+right+top+left
 V=lambda x,z:float(-E(z)[0]*x-E(z)[2]*z)
 jumpV=V(.3,0.)-float(-E1[0]*.3)
 assert abs(dflux-free*area)<2e-23 and abs(eflux-(free+bound)*area)<2e-23 and abs(line)<1e-12 and jumpV==0
 static.append(dict(epsilon1=eps1,epsilon2=eps2,normalDisplacement1=D1,freeSurfaceCharge=free,tangentialElectric=Et,electric1=E1.tolist(),electric2=E2.tolist(),polarization1=P1.tolist(),polarization2=P2.tolist(),boundSurfaceCharge=bound,pillboxRadius=R,pillboxArea=area,independentDFlux=dflux,independentVacuumEFlux=eflux,independentElectricCirculation=line,potentialJump=jumpV))
for e1,e2,m1,m2 in itertools.product([1.,2.,4.],repeat=4):
 eps1,eps2,mu1,mu2=eps0*e1,eps0*e2,mu0*m1,mu0*m2;Z1,Z2=math.sqrt(mu1/eps1),math.sqrt(mu2/eps2)
 # Solve boundary E/H matching, not a direct Fresnel formula.
 r,t=np.linalg.solve(np.array([[1.,-1.],[1/Z1,1/Z2]]),np.array([-1.,1/Z1]))
 Ei=lambda phase:math.cos(phase);Er=lambda phase:r*math.cos(phase);Et=lambda phase:t*math.cos(phase)
 avg=lambda f:quad(f,0,2*math.pi,epsabs=1e-14)[0]/(2*math.pi)
 Pi=avg(lambda phase:Ei(phase)**2/Z1);Pr=avg(lambda phase:-Er(phase)**2/Z1);Pt=avg(lambda phase:Et(phase)**2/Z2)
 boundary=lambda phase:(Ei(phase)+Er(phase))*(Ei(phase)-Er(phase))/Z1-Et(phase)**2/Z2
 error=avg(boundary);assert abs(error)<1e-17 and abs(1+Pr/Pi-Pt/Pi)<2e-14
 # r=0,t=1 preserves E but requires a current sheet when Z differs.
 Kx=1/Z1-1/Z2;missingPower=.5/Z1-.5/Z2;sheetWork=avg(lambda phase:Kx*math.cos(phase)**2)
 assert abs(missingPower-sheetWork)<1e-17
 waves.append(dict(epsilon1=eps1,epsilon2=eps2,mu1=mu1,mu2=mu2,impedance1=Z1,impedance2=Z2,independentReflectionAmplitude=float(r),independentTransmissionAmplitude=float(t),incidentAveragePower=Pi,reflectedSignedAveragePower=Pr,transmittedAveragePower=Pt,independentBoundaryPowerResidual=error,forcedZeroReflectionRequiredSheetCurrentX=Kx,forcedZeroReflectionSheetWork=sheetWork))
e1=eps0;e2=math.nextafter(e1,math.inf);Z1=math.sqrt(mu0/e1);Z2=math.sqrt(mu0/e2);naive=(Z2-Z1)/(Z2+Z1);stable=math.tanh(-.25*math.log1p((e2-e1)/e1));M,E1,E2=map(mp.mpf,[mu0,e1,e2]);A=mp.sqrt(M/E1);B=mp.sqrt(M/E2);ref=(B-A)/(B+A)
assert stable!=0 and abs(stable/float(ref)-1)<5e-15
negative=dict(tangentialJump=.3,rectangleSpan=.6,electricCirculation=-.3*.6,potentialJumpAtX03=-.3*.3,staticMaxwellCompatible=False)
result=dict(provenance='Actual vector pillbox surface and rectangle edge quadrature,independent NumPy E/H boundary solve,phase-averaged signed Poynting quadrature and180-digit exact-input near-match impedance reference. No product imports.',dielectricCases=static,waveCases=waves,nearMatchedImpedance=dict(epsilon1=e1,epsilon2=e2,mu=mu0,naiveReflection=naive,stableReflection=stable,highPrecisionReflection=float(ref)),tangentialJumpNegativeControl=negative,limitations=['Two nondispersive isotropic positive-coefficient half-spaces at rest,not all constitutive media.','Fixed normal +z from medium1 to medium2;side limits at interface,not a fabricated average.','Planar idealizations need remote sources;no finite isolated device claim.','An E-continuous,H-mismatched wave may be supported by a free surface-current sheet;it is only incompatible with the no-sheet request.','Nonmagnetic refractive-index formulas need their permeability assumptions.','Selected numerical observations are not certified global error bounds.'])

def derivative(f,x,h=1e-4):return(-f(x+2*h)+8*f(x+h)-8*f(x-h)+f(x-2*h))/(12*h)
spacetime=[]
for i,(e1,e2,m1,m2,k,z)in enumerate(itertools.product([.5,1.,2.],[.5,2.],[.5,1.],[1.,2.],[.5,1.2],[-.4,.4])):
 c1=1/math.sqrt(e1*m1);c2=1/math.sqrt(e2*m2);Z1=math.sqrt(m1/e1);Z2=math.sqrt(m2/e2);omega=k*c1;k2=omega/c2;t0=.3;A=(-1.)**i
 r,t=np.linalg.solve(np.array([[1.,-1.],[1/Z1,1/Z2]]),np.array([-1.,1/Z1]))
 E1=lambda z,time:A*(math.cos(k*z-omega*time)+r*math.cos(k*z+omega*time))
 H1=lambda z,time:A*(math.cos(k*z-omega*time)-r*math.cos(k*z+omega*time))/Z1
 E2=lambda z,time:A*t*math.cos(k2*z-omega*time)
 H2=lambda z,time:E2(z,time)/Z2
 sides=[]
 for E,H,e,m in[(E1,H1,e1,m1),(E2,H2,e2,m2)]:
  ev,hv=E(z,t0),H(z,t0)
  sides.append(dict(electricValue=ev,magneticIntensity=hv,energyDensity=.5*e*ev*ev+.5*m*hv*hv,poyntingZ=ev*hv,electricZ=derivative(lambda x:E(x,t0),z),electricT=derivative(lambda time:E(z,time),t0),magneticIntensityZ=derivative(lambda x:H(x,t0),z),magneticIntensityT=derivative(lambda time:H(z,time),t0)))
 spacetime.append(dict(permittivity1=e1,permittivity2=e2,permeability1=m1,permeability2=m2,amplitude=A,wavenumber1=k,z=z,time=t0,medium1=sides[0],medium2=sides[1]))
near=[]
delta=2.**-52
cases=[(eps0,math.nextafter(eps0,math.inf),mu0,mu0),(eps0,math.nextafter(eps0,0.),mu0,mu0),(1-delta,1.,1.,1+delta),(1.,1.,1.,1.),(1.,2.,1.,2.),(1e-12,4.,4.,1e-12),(4.,1e-12,1e-12,4.)]
for e1,e2,m1,m2 in cases:
 E1,E2,M1,M2=map(mp.mpf,[e1,e2,m1,m2]);Z1=mp.sqrt(M1/E1);Z2=mp.sqrt(M2/E2);R=(Z2-Z1)/(Z2+Z1);T=2*Z2/(Z1+Z2)
 near.append(dict(permittivity1=e1,permittivity2=e2,permeability1=m1,permeability2=m2,amplitude=1.,wavenumber1=1.,reflection=float(R),transmission=float(T),reflectance=float(R*R),transmittance=float(Z1/Z2*T*T)))
tiny=[]
for e1,e2,d,free in[(eps0,eps0,1e-9,1e-30),(eps0,math.nextafter(eps0,math.inf),2e-9,0.)]:
 E1,E2,D,Q=map(mp.mpf,[e1,e2,d,free]);n1=D/E1;n2=(D+Q)/E2;B=(E1-mp.mpf(eps0))*n1-(E2-mp.mpf(eps0))*n2
 tiny.append(dict(permittivity1=e1,permittivity2=e2,normalDisplacement1=d,freeSurfaceCharge=free,tangentialElectric=1.,normalEJump=float(n2-n1),boundSurfaceCharge=float(B)))
result['spacetimeCases']=spacetime;result['highPrecisionImpedanceCases']=near;result['tinyDielectricJumps']=tiny
result['provenance']+=' Plus independent spacetime five-point differences and180-digit near-cancelling product/charge references.'
target=Path(__file__).resolve().parents[1]/'fixtures/material-interfaces-independent.json';target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(dielectricCases=len(static),waveCases=len(waves),spacetimeCases=len(spacetime),highPrecisionImpedanceCases=len(near),tinyDielectricJumps=len(tiny))))
