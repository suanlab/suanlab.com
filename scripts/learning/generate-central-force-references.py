from pathlib import Path
from fractions import Fraction as Q
import random,json,math
import mpmath as mp
from scipy.integrate import solve_ivp
mp.mp.dps=70
rng=random.Random(20261104414)
radial=[];elliptic=[];roots=[]
special=[[1,2,0,10,10,1],[2,4,0,10,20,2],[2,2,0,20,10,1],[1,2,10,20,30,1],[1,2,0,0,30,1],[1,2,20,0,100,1],[1,2,0,10,20,1],[1,2,10,-10,10,4],[6,1,-30,1,120,4]]
for i in range(700):
 p=special[i] if i<len(special) else [rng.randint(1,6),rng.randint(1,8),rng.randint(-30,30),rng.randint(-30,30),rng.randint(1,120),rng.randint(1,4)]
 mu,rc,vrc,vtc,qc,m=p;r=Q(rc,2);vr=Q(vrc,10);vt=Q(vtc,10);q=Q(qc,10)
 # Eccentricity norm evaluated from the original Cartesian vector identity
 # e=(v^2/mu-1/r)q-(q.v/mu)v, independently of polar reduction.
 q0=[r,Q(0)];v=[vr,vt];v2=sum(x*x for x in v);qdotv=sum(x*y for x,y in zip(q0,v));ev=[(v2/mu-1/r)*x-qdotv/mu*y for x,y in zip(q0,v)]
 eps=v2/2-Q(mu)/r;h=r*vt;eff=-Q(mu)/q+h*h/(2*q*q);k=eps-eff
 radial.append(dict(values=p,exact={k:str(v)for k,v in dict(specificEnergy=eps,totalEnergy=m*eps,specificAngularMomentum=h,angularMomentum=m*h,eccentricitySquared=sum(x*x for x in ev)).items()},eVector=[str(x)for x in ev],query={k:str(v)for k,v in dict(physicalPotential=-Q(mu)/q,angularBarrier=h*h/(2*q*q),effectivePotential=eff,radialKineticPerMass=k,tangentialVelocity=h/q).items()},allowed=k>=0))
# Pure bisection at 70 decimal digits, no Newton step or TS dependency.
def root(M,e):
 if e==0:return M
 lo=M-e;hi=M+e
 for _ in range(260):
  z=(lo+hi)/2
  if z-e*mp.sin(z)>M:hi=z
  else:lo=z
 return (lo+hi)/2
for i in range(220):
 e=mp.mpf(rng.randint(0,19))/20;M=mp.mpf(rng.randint(-139000,139000))/1000
 if i<8:M,e=[(mp.mpf('1e-10'),mp.mpf('.95')),(mp.mpf('0'),mp.mpf('.95')),(mp.mpf('139'),mp.mpf('.95')),(mp.mpf('-139'),mp.mpf('.95')),(mp.pi,mp.mpf('.95')),(2*mp.pi,mp.mpf('.95')),(mp.mpf('1'),mp.mpf('0')),(mp.mpf('-1e-10'),mp.mpf('.95'))][i]
 if 8<=i<16:
  M,e=[(mp.mpf('1e-15'),mp.mpf('.999999')),(mp.mpf('-1e-15'),mp.mpf('.999999')),(mp.mpf('1e-25'),mp.mpf('0.9999999999999999')),(mp.mpf('-1e-25'),mp.mpf('0.9999999999999999')),(mp.mpf('1e6'),mp.mpf('.95')),(mp.mpf('-1e6'),mp.mpf('.95')),(mp.mpf('.1'),mp.mpf('.999999999999')),(mp.mpf('-.1'),mp.mpf('.999999999999'))][i-8]
 # Root references solve the exact binary values that will be parsed by JS.
 M,e=mp.mpf(float(M)),mp.mpf(float(e))
 roots.append(dict(M=float(M),e=float(e),E=float(root(M,e))))
special=[[1,2,0,0,0,10,1],[1,2,10,0,0,10,1],[6,1,19,0,0,200,1],[6,1,19,12,1,200,4],[1,8,19,-6,0,0,3],[2,4,0,6,1,200,2],[1,2,19,0,0,1,1]]
for i in range(160):
 p=special[i] if i<len(special) else [rng.randint(1,6),rng.randint(1,8),rng.randint(0,19),rng.randint(-12,12),rng.randint(0,1),rng.randint(0,200),rng.randint(1,4)]
 mu,ac,ec,oc,retro,tc,m=p;a=mp.mpf(ac)/2;e=mp.mpf(ec)/20;t=mp.mpf(tc)/10;direction=1-2*retro;n=mp.sqrt(mu/a**3);M=direction*n*t;E=root(M,e);b=a*mp.sqrt(1-e*e);theta=oc*mp.pi/12;c=mp.cos(theta);s=mp.sin(theta)
 rotate=lambda x,y:[float(c*x-s*y),float(s*x+c*y)]
 x=a*(mp.cos(E)-e);y=b*mp.sin(E);Edot=direction*n/(1-e*mp.cos(E));vx=-a*mp.sin(E)*Edot;vy=b*mp.cos(E)*Edot;r=mp.sqrt(x*x+y*y)
 exactRoot=dict(E=float(E),M=float(M),position=rotate(x,y),velocity=rotate(vx,vy),radius=float(r),energy=float(-m*mu/(2*a)),angularMomentum=float(m*direction*mp.sqrt(mu*a*(1-e*e))),area=float(a*b*(E-e*mp.sin(E))/2))
 # Integrate the inverse-square Cartesian force with a focus at the origin;
 # integrate area directly from cross(q,v)/2, no eccentric anomaly in ODE.
 r0=float(a*(1-e));vp=direction*math.sqrt(mu*float(1+e)/r0);cc=math.cos(float(theta));ss=math.sin(float(theta));initial=[r0*cc,r0*ss,-vp*ss,vp*cc,0.]
 def rhs(time,z):
  x,y,vx,vy,area=z;r=math.hypot(x,y)
  return[vx,vy,-mu*x/r**3,-mu*y/r**3,(x*vy-y*vx)/2]
 if t:
  sol=solve_ivp(rhs,(0,float(t)),initial,method='DOP853',rtol=2.3e-14,atol=1e-15)
  assert sol.success;z=list(map(float,sol.y[:,-1]))
 else:z=initial
 elliptic.append(dict(values=p,root=exactRoot,odePosition=z[:2],odeVelocity=z[2:4],odeArea=z[4],odeSteps=0 if not t else len(sol.t)))
 print('orbit',i,'steps',elliptic[-1]['odeSteps'],flush=True) if i%20==0 else None
Path('scripts/fixtures/central-force-independent.json').write_text(json.dumps(dict(radial=radial,elliptic=elliptic,roots=roots),separators=(',',':'))+'\n')
print('Generated 700 exact Cartesian-invariant references, 220 high-precision bisection roots and 160 independent Cartesian ODE references.',flush=True)
