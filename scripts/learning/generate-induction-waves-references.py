"""Independent references; NumPy/SciPy/mpmath required, no site imports.
Run from any directory. Observed differences are not certified error bounds.
"""
from pathlib import Path
import math,json,itertools
import numpy as np
import mpmath as mp
from scipy.integrate import quad
mp.mp.dps=160

def d(f,x,h=1e-4):return(-f(x+2*h)+8*f(x+h)-8*f(x-h)+f(x-2*h))/(12*h)
def d2(f,x,h=1e-3):return(-f(x+2*h)+16*f(x+h)-30*f(x)+16*f(x-h)-f(x-2*h))/(12*h*h)
loops=[];waves=[];negatives=[];tiny=[];flux_changes=[]
for i,(b0,b1,b2,u)in enumerate(itertools.product([-1.,0.,1.],[-.5,0.,.5],[-.2,0.,.2],[-.2,0.,.2])):
 t=.7;H=.8;eps=[.5,1.,2.][i%3];point=np.array([.2,-.3,.4])
 width=lambda time:1.2+u*time
 B=lambda time:b0+b1*time+.5*b2*time*time
 E=lambda p,time:np.array([.5*(b1+b2*time)*p[1],-.5*(b1+b2*time)*p[0],0.])
 J=lambda p:np.array([-.5*eps*b2*p[1],.5*eps*b2*p[0],0.])
 def edge_integrals(time):
  L=width(time);vertices=[np.array(p)for p in[(0.,0.,0.),(L,0.,0.),(L,H,0.),(0.,H,0.)]];es=[];ms=[]
  for j,p in enumerate(vertices):
   end=vertices[(j+1)%4];v=end-p
   es.append(quad(lambda a:float(E(p+a*v,time)@v),0,1,epsabs=1e-12)[0])
   # v=(u*x/L,0,0), v cross B=(0,-u*x*B/L,0).
   ms.append(quad(lambda a:float(np.array([0.,-u*(p+a*v)[0]*B(time)/L,0.])@v),0,1,epsabs=1e-12)[0])
  return es,ms
 es,ms=edge_integrals(t);rate=-d(lambda time:B(time)*width(time)*H,t)
 dtE=[d(lambda time:E(point,time)[j],t)for j in range(3)]
 curlE=d(lambda x:E(np.array([x,point[1],point[2]]),t)[1],point[0])-d(lambda y:E(np.array([point[0],y,point[2]]),t)[0],point[1])
 ta,tb=-.3,1.1
 integrated_e=quad(lambda time:sum(edge_integrals(time)[0]),ta,tb,epsabs=1e-12)[0]
 integrated_m=quad(lambda time:sum(edge_integrals(time)[1]),ta,tb,epsabs=1e-12)[0]
 loops.append(dict(magneticOffset=b0,magneticRate=b1,magneticAcceleration=b2,widthRate=u,initialWidth=1.2,height=H,permittivity=eps,permeability=1.,time=t,point=point.tolist(),electric=E(point,t).tolist(),magnetic=B(t),electricTimeDerivative=dtE,magneticTimeDerivative=d(B,t),electricCurlZ=curlE,conductionCurrent=J(point).tolist(),electricEdges=es,motionalEdges=ms,electricCirculation=sum(es),motionalCirculation=sum(ms),negativeFluxRate=rate,flux=B(t)*width(t)*H,startTime=ta,endTime=tb,integratedElectric=integrated_e,integratedMotional=integrated_m,fluxChange=B(tb)*width(tb)*H-B(ta)*width(ta)*H))
for c,A,width,beta in itertools.product([.5,1.,2.],[-1.,0.,1.],[.4,1.],[0.,.5,1.]):
 t=.31;z=.27;G=lambda x:A*math.exp(-x*x/(2*width*width))
 E=lambda z,t:beta*G(z-c*t)+(1-beta)*G(z+c*t)
 B=lambda z,t:(beta*G(z-c*t)-(1-beta)*G(z+c*t))/c
 ez=d(lambda x:E(x,t),z);bt=d(lambda x:B(z,x),t);bz=d(lambda x:B(x,t),z);et=d(lambda x:E(z,x),t)
 ezz=d2(lambda x:E(x,t),z);ett=d2(lambda x:E(z,x),t)
 span,z0,z1=.7,-.2,.8
 phi=lambda time:-span*quad(lambda x:B(x,time),z0,z1,epsabs=1e-12)[0]
 waves.append(dict(permittivity=1/(c*c),permeability=1.,amplitude=A,width=width,rightFraction=beta,z=z,time=t,electric=E(z,t),magnetic=B(z,t),electricZ=ez,magneticZ=bz,electricT=et,magneticT=bt,electricZZ=ezz,electricTT=ett,span=span,bottom=z0,top=z1,electricLine=span*(E(z0,t)-E(z1,t)),negativeMagneticFluxRate=-d(phi,t)))
for c,width in itertools.product([.5,1.,2.],[.4,.7,1.]):
 z,t=.7,.1;E=lambda z,t:math.exp(-(z-c*t)**2/(2*width*width))
 negatives.append(dict(permittivity=1/(c*c),permeability=1.,amplitude=1.,width=width,rightFraction=1.,z=z,time=t,magneticMode='zero-magnetic-negative-control',electricWaveResidual=d2(lambda x:E(z,x),t)-c*c*d2(lambda x:E(x,t),z),faradayResidual=d(lambda x:E(x,t),z),ampereResidual=-d(lambda x:E(z,x),t)/(c*c)))
for time in [1e-12,1e-16,1e-18,1e-50]:
 z,c,width=1.,1.,1.;Z,C,T,S=map(mp.mpf,[z,c,time,width]);gp=mp.exp(-(Z-C*T)**2/(2*S*S));gm=mp.exp(-(Z+C*T)**2/(2*S*S))
 tiny.append(dict(time=time,permittivity=1.,permeability=1.,amplitude=1.,width=width,rightFraction=.5,z=z,stableCB=float((gp-gm)/2),naiveCB=(math.exp(-(z-time)**2/2)-math.exp(-(z+time)**2/2))/2))
for rate,speed in[(1e-16,0.),(0.,1e-16)]:
 b0,L0,H,ta,tb=4.,1.,1.,0.,1.;B,R,L,U,HH,T0,T1=map(mp.mpf,[b0,rate,L0,speed,H,ta,tb]);reference=((B+R*T1)*(L+U*T1)-(B+R*T0)*(L+U*T0))*HH
 flux_changes.append(dict(magneticOffset=b0,magneticRate=rate,magneticAcceleration=0.,initialWidth=L0,widthRate=speed,height=H,startTime=ta,endTime=tb,fluxChange=float(reference)))
result=dict(provenance='Independent vector edge/time SciPy quadrature,spacetime five-point differences,and160-digit mpmath cancellation references; no product imports.',movingLoops=loops,transversePulses=waves,zeroBControls=negatives,tinyGaussianDifferences=tiny,tinyFluxChanges=flux_changes,limitations=['Prescribed bounded sources and loop geometry,not conductor dynamics or a finite isolated solenoid.','Plane pulses are infinite transversely; a scalar wave equation alone does not imply Maxwell compatibility.','Observed numerical differences are not certified error bounds.'])
target=Path(__file__).resolve().parents[1]/'fixtures/induction-waves-independent.json';target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:len(v)for k,v in result.items()if isinstance(v,list)}))
