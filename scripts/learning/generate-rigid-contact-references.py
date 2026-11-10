"""Independent Fraction linear systems/point sums and SciPy branch ODEs.
Run from repository root in a Python environment with scipy installed.
"""
from pathlib import Path
from fractions import Fraction as Q
import json,random,math
from scipy.integrate import solve_ivp
rng=random.Random(20261104418)
def solve(A,b):
 a=[[Q(x)for x in row]+[Q(y)]for row,y in zip(A,b)];n=len(a)
 for j in range(n):
  pivot=next(i for i in range(j,n)if a[i][j]);a[j],a[pivot]=a[pivot],a[j];d=a[j][j];a[j]=[x/d for x in a[j]]
  for i in range(n):
   if i!=j:
    v=a[i][j];a[i]=[x-v*y for x,y in zip(a[i],a[j])]
 return[row[-1]for row in a]
def sign(x):return(x>0)-(x<0)
def norm2(v):return sum((x*x for x in v),Q(0))
def cross(a,b):return a[0]*b[1]-a[1]*b[0]
def rod_sums(m,a,h,vx,vy,w):
 # Simpson quadrature exactly integrates the uniform rod's quadratic energy
 # and angular momentum, independently of a COM-inertia energy formula.
 P=[Q(0),Q(0)];L=Q(0);spin=Q(0);K=Q(0);Kcom=m*(vx*vx+vy*vy)/2
 for u,weight in [(Q(-1),m/6),(Q(0),2*m/3),(Q(1),m/6)]:
  r=[u*a,-u*h];v=[vx-w*r[1],vy+w*r[0]];mom=[weight*x for x in v];P=[x+y for x,y in zip(P,mom)];L+=cross([r[0]-a,r[1]+h],mom);spin+=cross(r,mom);K+=weight*norm2(v)/2
 return dict(velocity=[vx,vy],omega=w,contactVelocity=[vx+h*w,vy+a*w],momentum=P,spinAngularMomentum=spin,orbitalAngularMomentumAboutContact=L-spin,angularMomentumAboutContact=L,kineticTranslation=Kcom,kineticRotation=K-Kcom,kinetic=K)
def labels(d):return{k:[str(x)for x in v]if isinstance(v,list)else str(v)for k,v in d.items()}
rod=[];special=[[1,2,2,0,5,-10,5],[1,2,2,10,-10,0,10],[1,2,2,10,-10,0,0],[1,2,2,0,-10,10,5],[1,2,2,0,5,10,5],[4,0,6,-20,-20,20,10],[1,0,1,0,0,0,0]]
for i in range(900):
 p=special[i]if i<len(special)else[rng.randint(1,4),rng.randint(0,6),rng.randint(1,6),rng.randint(-20,20),rng.randint(-20,20),rng.randint(-20,20),rng.randint(0,10)]
 mc,ac,hc,vxc,vyc,wc,ec=p;m=Q(mc);a=Q(ac,2);h=Q(hc,2);vx=Q(vxc,10);vy=Q(vyc,10);w=Q(wc,10);e=Q(ec,10);I=sum(weight*((u*a)**2+(u*h)**2)for u,weight in [(Q(-1),m/6),(Q(0),2*m/3),(Q(1),m/6)]);c=vy+a*w;active=c<0
 dv,dw,j=solve([[m,0,-1],[0,I,-a],[1,a,0]],[0,0,-(1+e)*c])if active else[Q(0)]*3
 before=rod_sums(m,a,h,vx,vy,w);after=rod_sums(m,a,h,vx,vy+dv,w+dw)
 rod.append(dict(values=p,active=bool(active),impulse=str(j),before=labels(before),after=labels(after),loss=str(before['kinetic']-after['kinetic'])))

def abs_area(v0,v1,t):
 if v0*v1<0:
  split=t*abs(v0)/(abs(v0)+abs(v1));return(abs(v0)*split+abs(v1)*(t-split))/2
 return t*(abs(v0)+abs(v1))/2
wheel=[];odes=[]
special=[[1,2,2,2,5,30,0,0,0,10],[1,2,2,2,5,0,0,10,0,10],[1,2,2,2,5,0,30,0,0,10],[1,2,1,0,5,0,30,0,0,10],[1,2,2,0,5,100,0,-10,0,10],[4,2,1,0,10,79,0,20,0,100],[1,2,2,2,5,0,0,12,0,2],[1,1,0,0,1,-100,100,20,-20,100],[1,2,0,0,5,10,5,10,10,100],[1,2,1,0,5,30,0,0,0,10],[1,2,1,0,5,31,0,0,0,10]]
for i in range(1000):
 p=special[i]if i<len(special)else[rng.randint(1,4),rng.randint(1,6),rng.randint(0,10),rng.randint(0,10),rng.randint(1,10),rng.randint(-100,100),rng.randint(-100,100),rng.randint(-20,20),rng.randint(-20,20),rng.randint(0,100)]
 mc,rc,kc,dc,kappac,fc,tc,vc,wc,timec=p;m=Q(mc);R=Q(rc,2);kap=Q(kappac,10);I=m*R*R*kap;F=Q(fc,10);tau=Q(tc,10);V0=Q(vc,10);O0=Q(wc,10);T=Q(timec,10);u0=V0-R*O0;fk=Q(mc*kc);fs=Q(mc*(kc+dc))
 ar,br,required=solve([[m,0,-1],[0,I,R],[1,-R,0]],[F,tau,0]);can=abs(required)<=fs
 freeA,freeB,_=solve([[m,0,-1],[0,I,R],[0,0,1]],[F,tau,0]);freeU=freeA-R*freeB
 postF=required if can else-fk*sign(freeU);preF=postF if not u0 else-fk*sign(u0)
 def accel(f):return solve([[m,0,-1],[0,I,R],[0,0,1]],[F,tau,f])[:2]
 aa,bb=accel(preF);du=aa-R*bb;stop=-u0/du if u0 and u0*du<0 else None
 segments=[(T,preF)]if stop is None or T<stop else[(stop,preF),(T-stop,postF)]
 x=Q(0);phi=Q(0);V=V0;Om=O0;S=Q(0);D=Q(0);Wft=Q(0);Wfr=Q(0);Jf=Q(0)
 for dt,f in segments:
  aa,bb=accel(f);Vp=V+aa*dt;Op=Om+bb*dt;dx=(V+Vp)*dt/2;dp=(Om+Op)*dt/2;S+=abs_area(V-R*Om,Vp-R*Op,dt);D+=abs_area(V,Vp,dt);x+=dx;phi+=dp;Wft+=f*dx;Wfr-=R*f*dp;Jf+=f*dt;V,Om=Vp,Op
 f=postF if stop is not None and T>=stop else preF;aa,bb=accel(f);Ktrans=m*V*V/2;Krot=I*Om*Om/2;heat=-Wft-Wfr;rolling=(not u0 or(stop is not None and T>=stop))and can
 exact=labels(dict(position=x,clockwiseAngle=phi,velocity=V,clockwiseOmega=Om,slipVelocity=V-R*Om,friction=f,linearAcceleration=aa,angularAcceleration=bb,slipAcceleration=aa-R*bb,COMdistance=D,slipDistance=S,signedSlipDisplacement=x-R*phi,kineticTranslation=Ktrans,kineticRotation=Krot,kinetic=Ktrans+Krot,heat=heat,forceWork=F*x,torqueWork=tau*phi,externalWork=F*x+tau*phi,frictionImpulse=Jf,frictionTranslationWork=Wft,frictionRotationWork=Wfr,contactWork=Wft+Wfr,contactPower=f*(V-R*Om),heatRate=fk*abs(V-R*Om)))
 wheel.append(dict(values=p,query=exact,required=str(required),event=None if stop is None else str(stop),rolling=bool(rolling),eventReached=bool(stop is not None and T>=stop),feasible=bool(can)))
 if i<180:
  # Independently integrate actual branch dynamics, detecting contact slip0
  # with a terminal ODE event instead of evaluating the query polynomial.
  # On each valid branch the slip sign is known. Integrate that smooth
  # signed derivative through event interpolation; abs(u) outside the
  # pre-event domain would introduce an artificial cusp in dense output.
  def rhs(ff,slip_sign):
   def fun(t,z):
    xx,ph,v,om=z[:4];u=v-float(R)*om
    return[v,om,float((F+ff)/m),float((tau-R*ff)/I),float(F)*v+float(tau)*om,float(ff)*v,-float(R*ff)*om,float(slip_sign)*u,abs(v)]
   return fun
  initial=[0.,0.,float(V0),float(O0),0.,0.,0.,0.,0.];z=initial;found=None;preSign=sign(u0) if u0 else(0 if can else sign(freeU));postSign=0 if can else sign(freeU)
  if T:
   if u0 and stop is not None:
    def event(t,z):return z[2]-float(R)*z[3]
    event.terminal=True;event.direction=-sign(u0)
    sol=solve_ivp(rhs(preF,preSign),(0,float(T)),initial,events=event,method='DOP853',rtol=2e-12,atol=1e-14);assert sol.success;z=sol.y[:,-1]
    if len(sol.t_events[0]):
     found=float(sol.t_events[0][0]);remaining=float(T)-found
     if remaining>1e-13:
      sol2=solve_ivp(rhs(postF,postSign),(found,float(T)),z,method='DOP853',rtol=2e-12,atol=1e-14);assert sol2.success;z=sol2.y[:,-1]
   else:
    sol=solve_ivp(rhs(preF,preSign),(0,float(T)),initial,method='DOP853',rtol=2e-12,atol=1e-14);assert sol.success;z=sol.y[:,-1]
  odes.append(dict(values=p,position=float(z[0]),angle=float(z[1]),velocity=float(z[2]),omega=float(z[3]),externalWork=float(z[4]),frictionTranslationWork=float(z[5]),frictionRotationWork=float(z[6]),slipDistance=float(z[7]),COMdistance=float(z[8]),event=found))
Path('scripts/fixtures/rigid-contact-independent.json').write_text(json.dumps(dict(rod=rod,wheel=wheel,ode=odes),separators=(',',':'))+'\n');print('Generated900 exact rod linear-system/quadrature cases,1000 exact wheel branch/point-work cases and180 independent event-ODE references.')
