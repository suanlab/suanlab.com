#!/usr/bin/env python3
"""Independent passive relaxation / Fourier aperture checks; standard library only.
No product imports. Finite quadrature checks complement the explicit manuscript proofs.
"""
from fractions import Fraction as F
from collections import Counter
import math,cmath,json,sys

def relaxation(ci,dc,r,tau,vs,vi,pi,t):
 if not all(math.isfinite(x) for x in [ci,dc,r,tau,vs,vi,pi,t]) or min(ci,dc,r,tau)<=0 or t<0:raise ValueError('Positive Cinf, DeltaC, R, tau and nonnegative time required')
 eta=1/(r*ci);rho=1/tau;beta=dc/ci;total=eta+(1+beta)*rho
 disc=(eta-rho)**2+2*beta*rho*(eta+rho)+beta*beta*rho*rho
 root=math.sqrt(disc);fast=(total+root)/2;slow=2*eta*rho/(total+root)
 x=vi-vs;y=pi-dc*vs;xdot=-(eta+beta*rho)*x+rho*y/ci
 xs=(xdot+fast*x)/(fast-slow);xf=x-xs
 modes=[(xs,slow),(xf,fast)]
 v=vs+sum(a*math.exp(-rate*t) for a,rate in modes)
 p=dc*vs+sum(ci/rho*(eta+beta*rho-rate)*a*math.exp(-rate*t) for a,rate in modes)
 return v,p,slow,fast

def aperture(q):
 if not math.isfinite(q) or abs(q)>8:raise ValueError('Finite |q| <= 8 for this bounded series verifier')
 term=1.;terms=[term]
 for m in range(1,70):term*=-q*q/(4*m*(m+1));terms.append(term)
 return math.fsum(terms)

def rational_aperture(q,n=60):
 """Alternating series at |q|<=4: return exact partial sum and next-term tail bound."""
 q=F(q)
 if abs(q)>4:raise ValueError('Exact tail bound restricted to |q| <=4')
 term=F(1);total=term
 for m in range(1,n+1):term*=-q*q/(4*m*(m+1));total+=term
 next_term=-term*q*q/(4*(n+1)*(n+2))
 assert abs(q*q/(4*(n+2)*(n+3)))<1
 return total,abs(next_term)

def rk4(ci,dc,r,tau,vs,vi,pi,t,n):
 def rhs(y):
  v,p=y[:2];i=(vs-v)/r;pd=(dc*v-p)/tau;vd=(i-pd)/ci
  return [vd,pd,vs*i,r*i*i,tau*pd*pd/dc]
 y=[vi,pi,0.,0.,0.];h=t/n
 for _ in range(n):
  a=rhs(y);b=rhs([x+h*k/2 for x,k in zip(y,a)]);c=rhs([x+h*k/2 for x,k in zip(y,b)]);d=rhs([x+h*k for x,k in zip(y,c)])
  y=[x+h*(aa+2*bb+2*cc+dd)/6 for x,aa,bb,cc,dd in zip(y,a,b,c,d)]
 return y

def close(a,b,tol=3e-9):assert abs(a-b)<=tol*max(1.,abs(b)),(a,b)
def verify():
 out=Counter()
 for ci in [F(1,2),F(1),F(2)]:
  for dc in [F(1,2),F(1),F(3)]:
   for tau in [F(1,10),F(1),F(5)]:
    for v,p,vs in [(F(0),F(1),F(1)),(F(1),F(0),F(0)),(F(-1),F(2),F(3)),(F(1),F(3),F(1))]:
     r=F(2);i=(vs-v)/r;pd=(dc*v-p)/tau;vd=(i-pd)/ci;ud=ci*v*vd+p*pd/dc;loss=tau*pd*pd/dc
     assert vs*i==r*i*i+ud+loss and loss>=0;out['exactInstantaneousPowerBalances']+=1
     veq=(ci*v+p)/(ci+dc);u0=ci*v*v/2+p*p/(2*dc);ueq=(ci+dc)*veq*veq/2
     extra=ci*dc/(2*(ci+dc))*(v-p/dc)**2
     assert ci*v+p==(ci+dc)*veq and u0-ueq==extra and extra>=0;out['exactInitialRedistributions']+=1
     eta=1/(r*ci);rho=1/tau;beta=dc/ci;total=eta+(1+beta)*rho
     disc=(eta-rho)**2+2*beta*rho*(eta+rho)+beta*beta*rho*rho
     assert disc==total*total-4*eta*rho and disc>0 and total>0;out['positiveDistinctRateProofCases']+=1
    for omega in [F(0),F(1,10),F(1),F(10)]:
     w=float(omega);y=-1j*w*float(ci)+(-1j*w*float(dc))/(1-1j*w*float(tau));real=dc*tau*omega*omega/(1+tau*tau*omega*omega)
     close(y.real,float(real),1e-12);assert real>=0;out['admittancePassivityCases']+=1
 # Independent integration of two states and three power histories.
 for ci,dc in [(.5,.5),(1.,2.)]:
  for tau in [.1,1.,4.]:
   for vs,vi,pi in [(0.,1.,0.),(1.,0.,0.),(-1.,2.,-.5),(1.,1.,dc)]:
    r=2.;t=4.;v,p,slow,fast=relaxation(ci,dc,r,tau,vs,vi,pi,t)
    assert 0<slow<fast;close(slow*fast,1/(r*ci*tau));out['modalRateProducts']+=1
    coarse=rk4(ci,dc,r,tau,vs,vi,pi,t,3200);fine=rk4(ci,dc,r,tau,vs,vi,pi,t,6400)
    close(fine[0],v,2e-9);close(fine[1],p,2e-9);out['independentStateIntegrations']+=1
    initial=ci*vi*vi/2+pi*pi/(2*dc);final=ci*v*v/2+p*p/(2*dc)
    defect=abs(fine[2]-fine[3]-fine[4]-(final-initial));old=abs(coarse[2]-coarse[3]-coarse[4]-(final-initial))
    assert fine[3]>=0 and fine[4]>=0;assert defect<2e-8*max(1,abs(final-initial));assert defect<=old*1.2+1e-12;out['integratedEnergyBudgets']+=1
 for vi,pi in [(1.,0.),(0.,1.),(-1.,2.),(1.,1.)]:
  after=(vi+pi)/2;outer=after*math.exp(-.2/2);wrong=vi*math.exp(-.2/2)
  errors=[]
  for tau in [.01,.002,.0004]:
   v,p,_,_=relaxation(1.,1.,1.,tau,0.,vi,pi,.2);errors.append(abs(v-outer));out['fastLayerOuterComparisons']+=1
  assert errors[-1]<=errors[0]+1e-12
  if vi!=pi:assert abs(wrong-outer)>.1 and errors[-1]<abs(wrong-outer)/100;out['unpreparedInitialCounterexamples']+=1
 for bad in [(0,1,1,1),(1,0,1,1),(1,1,0,1),(1,1,1,0)]:
  try:relaxation(*bad,0,1,0,1)
  except ValueError:out['invalidRelaxationRejections']+=1
  else:raise AssertionError('singular model accepted')
 # Exact series tails certify a sign-changing bracket, not an exact double-precision zero.
 lo=F('3.8317059702075123156144');hi=F('3.8317059702075123156145');a,ea=rational_aperture(lo);b,eb=rational_aperture(hi)
 assert a-ea>0 and b+eb<0;out['certifiedRootBrackets']+=1
 assert hi<4 and 1-hi*hi/32>F(1,2);out['exactHalfRadiusLowerBounds']+=1
 rounded=float((lo+hi)/2);assert F.from_float(rounded)<lo or F.from_float(rounded)>hi
 val,err=rational_aperture(F.from_float(rounded));assert abs(val)>err;out['roundedRootIsNotExactZero']+=1
 for q in [0.,.125,.5,1.,2.,3.,4.,6.,8.]:
  # Midpoint angle average of a circular rim plus radial Simpson integration
  # A disk can also be integrated in Cartesian strips: 4/pi int_0^1 sqrt(1-x²) cos(qx) dx.
  # Use polar disk coordinates with Simpson in normalized radius and independent angular midpoint.
  nang=256;nr=128;dr=1/nr
  def rim(rho):return sum(math.cos(q*rho*math.cos(2*math.pi*(j+.5)/nang)) for j in range(nang))/nang
  value=sum((1 if i in [0,nr] else 4 if i%2 else 2)*2*(i*dr)*rim(i*dr) for i in range(nr+1))*dr/3
  close(value,aperture(q),2e-8);out['independentDiskQuadratures']+=1
  if q<=4:
   exact,tail=rational_aperture(F.from_float(q));close(aperture(q),float(exact),3e-13);assert tail<F(1,10**100);out['fractionSeriesTailChecks']+=1
 for q in [F(0),F(1,8),F(1,4),F(1,2),F(1)]:
  exact,tail=rational_aperture(q);lower=1-q*q/8;upper=lower+q**4/192
  assert lower<=exact-tail if q else exact==1
  assert exact+tail<=upper if q else exact==1;out['smallApertureErrorBounds']+=1
 # Compare oriented E and v-cross-B line integrals with differentiated disk flux.
 for k in [.5,1.,2.,4.]:
  for radius in [.25,.5,1.]:
   for omega,v in [(0.,.5),(1.,0.),(1.,.5),(1.,1.)]:
    phase=.7;n=2048;da=2*math.pi/n;factor=radius*sum(math.cos(2*math.pi*(j+.5)/n)*cmath.exp(1j*k*radius*math.cos(2*math.pi*(j+.5)/n)) for j in range(n))*da
    electric=omega/k*cmath.exp(1j*phase)*factor;motion=-v*cmath.exp(1j*phase)*factor
    expected=1j*(omega-k*v)*math.pi*radius*radius*aperture(k*radius)*cmath.exp(1j*phase)
    assert abs(electric+motion-expected)<2e-12;out['independentMovingLineIntegrals']+=1
    if omega==k*v:assert abs(electric+motion)<2e-12;out['kinematicBlindCases']+=1
 # Radius diversity at the ideal aperture zero; bounded bracket carries uncertainty.
 root=(float(lo)+float(hi))/2;half=aperture(root/2);assert half>.5
 for omega in [.5,1.,2.]:
  c=[0j,1j*omega*.25*half]
  for b in [1+0j,.5+.3j,-1j]:
   y=[x*b for x in c];den=sum(abs(x)**2 for x in c);estimate=sum(x.conjugate()*z for x,z in zip(c,y))/den;close(abs(estimate-b),0,1e-13);out['idealNullRadiusDiversitySolves']+=1
   for noise in [(.01+0j,0j),(0j,.01j),(.005+.003j,-.002j)]:
    noisy=sum(x.conjugate()*(z+e) for x,z,e in zip(c,y,noise))/den
    assert abs(noisy-b)<=math.sqrt(sum(abs(e)**2 for e in noise)/den)+1e-13;out['conditionalInverseNoiseBounds']+=1
 for q in [float(lo),float(hi)]:
  assert math.isclose(aperture(q/2),half,rel_tol=2e-15);out['rootBracketDiversityChecks']+=1
 return dict(sorted(out.items()))

def references():
 rc=[]
 for r in [1.,2.]:
  for vi in [-1.,0.,1.]:
   for theta in [0.,.25,1.,4.]:
    c=math.pi;t=theta*r*c;vs=1.;v=vs+(vi-vs)*math.exp(-theta)
    finite=[]
    for ratio in [.01,.001,.0001]:
     value=relaxation(.4*c,.6*c,r,ratio*r*c,vs,vi,.6*c*vi,t)[0];finite.append(value)
    rc.append(dict(R=r,vi=vi,theta=theta,voltage=v,relaxingVoltages=finite))
 coils=[]
 phase=.7
 for k in [.25,.5,1.]:
  for radius in [.25,.5,1.]:
   for b in [-1.,.5]:
    q=k*radius;coils.append(dict(k=k,radius=radius,b=b,phase=phase,B0=b*math.cos(phase),B0t=b*math.sin(phase),g=-k*k*b*math.cos(phase)/4,gt=-k*k*b*math.sin(phase)/4,quadratic=1-q*q/8,transfer=aperture(q),bound=q**4/192))
 return dict(rc=rc,coils=coils)
if __name__=='__main__':print(json.dumps(references() if '--references' in sys.argv else verify(),sort_keys=True))
