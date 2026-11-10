"""Independent IVP, energy/area/edge quadrature, field differences and exact-input references.
Run: python3 scripts/learning/generate-circuit-sensors-references.py
Requires numpy, scipy, mpmath. No product imports. Selected observations,not global error bounds.
"""
from pathlib import Path
import math,itertools,json
import numpy as np
from scipy.integrate import solve_ivp,quad
import mpmath as mp
mp.mp.dps=180
mu0=4*math.pi*1e-7;eps0=1/(mu0*299792458**2)
rc=[]
for R,eps,radius,gap,vi,theta in itertools.product([1.,2.,1e5],[eps0,1.],[.2,.7],[.05,.2],[-.4,.2],[.3,1.2]):
 mu=mu0;A=math.pi*radius**2;C=eps*A/gap;T=R*C;vs=1.;f=.6;queryR=f*radius
 sol=solve_ivp(lambda t,V:[vs-V[0]],[0,theta],[vi],rtol=2e-12,atol=2e-14)
 V=lambda tau:vs+(vi-vs)*math.exp(-tau);Vp=lambda tau:(vs-vi)*math.exp(-tau)/T;I=lambda tau:C*Vp(tau);B=lambda tau,r:.5*mu*eps*r*Vp(tau)/gap
 energy=quad(lambda tau:(vs*I(tau)-R*I(tau)**2)*T,0,theta,epsabs=1e-13)[0]
 stored=.5*C*(V(theta)**2-vi**2);h=1e-4
 faraday=(B(theta+h,queryR)-B(theta-h,queryR))/(2*h*T)
 # A radial derivative of actual B yields curl Bz;compare displacement contribution.
 dh=radius*1e-4;curlB=((queryR+dh)*B(theta,queryR+dh)-(queryR-dh)*B(theta,queryR-dh))/(2*dh*queryR)
 displacement=mu*eps*(V(theta+h)-V(theta-h))/(2*h*T*gap)
 magneticEnergy=quad(lambda r:B(theta,r)**2/(2*mu)*2*math.pi*r*gap,0,radius,epsabs=1e-13)[0]
 assert abs(sol.y[0,-1]-V(theta))<5e-12 and abs(energy-stored)<2e-12*max(1,abs(stored))
 rc.append(dict(R=R,epsilon=eps,mu=mu,radius=radius,gap=gap,initialV=vi,theta=theta,radiusFraction=f,voltage=float(sol.y[0,-1]),current=I(theta),netWork=energy,storedChange=stored,faradayPhi=faraday,curlBz=curlB,muDisplacement=displacement,gapMagneticEnergy=magneticEnergy))
# Genuine fields and distributed current,with mixed derivatives evaluated from actual E/B.
def field(x,y,t):
 B0=.8+.3*t-.1*t*t;g=.4+.2*t+.05*t*t;B0t=.3-.2*t;gt=.2+.1*t;r2=x*x+y*y;f=.5*B0t+.25*gt*r2
 return np.array([f*y,-f*x,0.]),np.array([0.,0.,B0+g*r2])
fieldCases=[]
for x,y,t in itertools.product([-.7,0.,.4],[-.5,0.,.6],[-.3,.2,.8]):
 h=1e-4;E,B=field(x,y,t);Ex_p,Bx_p=field(x+h,y,t);Ex_m,Bx_m=field(x-h,y,t);Ey_p,By_p=field(x,y+h,t);Ey_m,By_m=field(x,y-h,t);Et_p,Bt_p=field(x,y,t+h);Et_m,Bt_m=field(x,y,t-h)
 curlEz=(Ex_p[1]-Ex_m[1]-Ey_p[0]+Ey_m[0])/(2*h);Bt=(Bt_p[2]-Bt_m[2])/(2*h)
 J=np.array([(By_p[2]-By_m[2])/(2*h),-(Bx_p[2]-Bx_m[2])/(2*h),0.])/mu0-eps0*(Et_p-Et_m)/(2*h)
 assert abs(curlEz+Bt)<2e-8
 fieldCases.append(dict(x=x,y=y,time=t,electric=E.tolist(),magnetic=B.tolist(),current=J.tolist(),curlElectricZ=curlEz,magneticTimeRate=Bt))
coils=[]
for radius,ox,oy,t in itertools.product([.3,.7,1.1],[-.5,.4],[-.2,.6],[.2,.8]):
 vx,vy=.17,-.09;turns=3
 def flux(t):
  x0=ox+vx*t;y0=oy+vy*t
  return quad(lambda th:quad(lambda r:field(x0+r*math.cos(th),y0+r*math.sin(th),t)[1][2]*r,0,radius,epsabs=1e-11)[0],0,2*math.pi,epsabs=1e-11)[0]
 def line(th,moving):
  x=ox+vx*t+radius*math.cos(th);y=oy+vy*t+radius*math.sin(th);E,B=field(x,y,t);F=np.cross(np.array([vx,vy,0.]),B)if moving else E;return float(F@np.array([-radius*math.sin(th),radius*math.cos(th),0.]))
 electric=turns*quad(lambda th:line(th,False),0,2*math.pi,epsabs=1e-12)[0];motional=turns*quad(lambda th:line(th,True),0,2*math.pi,epsabs=1e-12)[0];h=1e-3
 derivative=turns*(-flux(t+2*h)+8*flux(t+h)-8*flux(t-h)+flux(t-2*h))/(12*h)
 assert abs(derivative+electric+motional)<1e-10
 coils.append(dict(radius=radius,center=[ox,oy],velocity=[vx,vy],turns=turns,time=t,fluxPerTurn=flux(t),transformerEmf=electric,motionalEmf=motional,fluxLinkageRate=derivative))
# Fixed-coil voltage matrix uses rates,not absolute fields.
inference=[]
for radii in [[.3,.7],[.7,.70000001]]:
 centers=[[.1,-.2],[-.3,.4]]if radii==[.3,.7]else[[0.,0.],[0.,0.]];true=np.array([.3,.2]);M=np.array([[1,x*x+y*y+r*r/2]for r,(x,y)in zip(radii,centers)]);turns=[2,3];voltage=[]
 for r,(x0,y0),N in zip(radii,centers,turns):
  voltage.append(N*quad(lambda th:float(field(x0+r*math.cos(th),y0+r*math.sin(th),0.)[0]@np.array([-r*math.sin(th),r*math.cos(th),0.])),0,2*math.pi,epsabs=1e-12)[0])
 y=-np.array(voltage)/np.array([N*math.pi*r*r for N,r in zip(turns,radii)]);estimate=np.linalg.solve(M,y)
 gains=[float(N*(math.pi*r*r))for N,r in zip(turns,radii)];S=[sum(mp.mpf(x)**2 for x in center)+mp.mpf(r)**2/2 for r,center in zip(radii,centers)];E=list(map(mp.mpf,voltage));G=list(map(mp.mpf,gains));slope=(E[0]*G[1]-E[1]*G[0])/(G[0]*G[1]*(S[1]-S[0]));intercept=-E[0]/G[0]-S[0]*slope
 inference.append(dict(radii=radii,centers=centers,turns=turns,voltages=voltage,trueRates=true.tolist(),independentLinearSolve=estimate.tolist(),highPrecisionRoundedGainRates=[float(intercept),float(slope)],highPrecisionRateStrings=[str(intercept),str(slope)],conditionInfinity=float(np.linalg.cond(M,np.inf))))
geometry=[]
for r1,r2,c1,c2 in [(.3,math.nextafter(.3,math.inf),[4.,4.],[4.,4.]),(.7,.70000001,[0.,0.],[0.,0.]),(.5,.5,[1.,0.],[0.,1.]),(5e-324,1e-323,[0.,0.],[0.,0.])]:
 ref=sum(mp.mpf(x)**2 for x in c2)-sum(mp.mpf(x)**2 for x in c1)+(mp.mpf(r2)**2-mp.mpf(r1)**2)/2
 geometry.append(dict(radii=[r1,r2],centers=[c1,c2],exact=str(ref),expected=float(ref)if abs(ref)>=mp.mpf(2)**-1074 else None,exactZero=ref==0))
tiny=[]
for theta in [1e-20,1e-12,.3]:
 ref=-mp.expm1(-mp.mpf(theta));tiny.append(dict(theta=theta,initialV=1.,supplyV=2.,expectedChange=float(ref),exact=str(ref)))
result=dict(provenance='No product imports. SciPy IVP and signed energy/area/edge quadrature;actual field differences;NumPy measured-voltage solve;180-digit exact binary-input geometry and exponential changes.',rcCases=rc,fieldCases=fieldCases,coilCases=coils,inferenceCases=inference,exactGeometryContrasts=geometry,tinyRcChanges=tiny)
p=Path('scripts/fixtures/circuit-sensors-independent.json');p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:len(v)for k,v in result.items()if isinstance(v,list)}))
