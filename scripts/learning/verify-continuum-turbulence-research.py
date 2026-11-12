"""Independent physical-field quadratures for the two stated turbulence examples.

Standard-library Gaussian/Legendre integration and exact rational counterexamples
supplement the analytic derivations; they do not validate a general RANS/LES model.
"""
from math import sin,cos,pi,sqrt,exp,isclose
from fractions import Fraction as F
import json

def gauss(n,left,right):
 pairs=[]
 for i in range(1,n+1):
  x=cos(pi*(i-.25)/(n+.5))
  for _ in range(30):
   p0,p1=1.,x
   for j in range(2,n+1):p0,p1=p1,((2*j-1)*x*p1-(j-1)*p0)/j
   derivative=n*(x*p1-p0)/(x*x-1);step=p1/derivative;x-=step
   if abs(step)<2e-16:break
  # Recompute derivative at the final root.
  p0,p1=1.,x
  for j in range(2,n+1):p0,p1=p1,((2*j-1)*x*p1-(j-1)*p0)/j
  derivative=n*(x*p1-p0)/(x*x-1)
  pairs.append(((left+right)/2+(right-left)*x/2,(right-left)/((1-x*x)*derivative*derivative)))
 return pairs

def poly(c,x,n=0):
 for _ in range(n):c=[(i+1)*v for i,v in enumerate(c[1:])]
 return sum(v*x**i for i,v in enumerate(c))
phi=[0,0,1,-2,1];chi=[0,0,-1,4,-5,2]
physical=0;negative=0;inviscid=0
for H in [.5,2.]:
 for b in [-.7,.7]:
  for c in [-.25,0.,.25]:
   for mu in [0.,.001,.1]:
    k,rho,ub,a,W=1.5,1.,.8,1.,1.;LW=2*pi/k*W;tot=[0.]*6
    for x,weight in gauss(16,0.,1.):
     Ucoeff=[0,6*ub*(1-c),6*ub*(3*c-1),-12*ub*c]
     U=poly(Ucoeff,x);Uy=poly(Ucoeff,x,1)/H;Uyy=poly(Ucoeff,x,2)/(H*H)
     ph,ph1,ph2,ph3=[poly(phi,x,n)/(H**n) for n in range(4)]
     ch,ch1,ch2,ch3=[poly(chi,x,n)/(H**n) for n in range(4)]
     for j in range(8):
      sn,cs=sin(2*pi*j/8),cos(2*pi*j/8)
      up=a*ph1*sn+b*ch1*cs;u=U+up;v=-k*a*ph*cs+k*b*ch*sn
      ux=k*(a*ph1*cs-b*ch1*sn);uy=Uy+a*ph2*sn+b*ch2*cs;vx=k*k*(a*ph*sn+b*ch*cs);vy=-ux
      lapu=-k*k*up+Uyy+a*ph3*sn+b*ch3*cs;lapv=-k*k*v-k*a*ph2*cs+k*b*ch2*sn
      fx=rho*(u*ux+v*uy)-mu*lapu;fy=rho*(u*vx+v*vy)-mu*lapv
      values=[U*fx,up*fx+v*fy,u*fx+v*fy,mu*Uy*Uy,mu*(ux*ux+(uy-Uy)**2+vx*vx+vy*vy),-rho*up*v*Uy]
      tot=[z+weight*H*LW*v/8 for z,v in zip(tot,values)]
    fm,ff,ft,dm,df,power=tot
    exactDm=mu*LW*ub*ub/H*(12+36*c*c/5)
    exactDf=mu*LW*(a*a*(H*k**4/1260+2*k*k/(105*H)+2/(5*H**3))+b*b*(H*k**4/13860+2*k*k/(315*H)+2/(7*H**3)))
    exactP=LW*4*rho*a*b*k*ub*c/(1155*H)
    for actual,expected in [(dm,exactDm),(df,exactDf),(power,exactP),(fm,dm+power),(ff,df-power),(ft,dm+df),(ft,fm+ff)]:
     assert isclose(actual,expected,rel_tol=3e-11,abs_tol=3e-12),(H,b,c,mu,actual,expected)
    assert dm>=0 and df>=0
    if ff<-1e-10:negative+=1
    if mu==0:assert dm==df==0;inviscid+=1
    physical+=1
assert negative>0
# Actual Gaussian displacement integration on [-9,9]; bounded sine fields have
# omitted normal tails below 3e-19, separate from finite quadrature error.
normal=[(x,w*exp(-x*x/2)/sqrt(2*pi)) for x,w in gauss(96,-9.,9.)]
def G(f,x,width):
 if width==0:return f(x)
 return sum(w*f(x+width*z) for z,w in normal)
filters=0;zero=0
for ell,delta in [(0.,.25),(.25,0.),(.25,.25),(.5,.75),(.75,.5)]:
 for A,B,C in [(1.,-.5,0.),(1.,.5,.7),(.3,-.2,-.7)]:
  sigma=sqrt(ell*ell+delta*delta)
  def u(x):return C+A*sin(x)+B*sin(2*x)
  def w(x):return C+A*exp(-ell*ell/2)*sin(x)+B*exp(-2*ell*ell)*sin(2*x)
  def v(x):return C+A*exp(-sigma*sigma/2)*sin(x)+B*exp(-2*sigma*sigma)*sin(2*x)
  def tau(x):return G(lambda xx:u(xx)**2,x,ell)-G(u,x,ell)**2
  for x in [.1,.7,1.9,3.]:
   vv=G(w,x,delta);assert abs(vv-v(x))<3e-12 and abs(G(u,x,ell)-w(x))<3e-12
   total=G(lambda xx:u(xx)**2,x,sigma)-vv*vv
   inherited=G(tau,x,delta);leonard=G(lambda xx:w(xx)**2,x,delta)-vv*vv
   assert abs(total-inherited-leonard)<3e-11
   assert min(total,inherited,leonard)>-3e-11
   if ell==0:assert abs(inherited)<3e-11
   if delta==0:assert leonard==0
   filters+=1
  if ell==0 or delta==0:zero+=1
p,q=F(3,4),F(2,3)
fine=(p*p-p**8)/8;coarse=((p*q)**2-(p*q)**8)/8
assert fine-coarse==F(14175,524288)>0
print(json.dumps(dict(actualEightPhaseForceBudgetCases=physical,negativeFluctuationForcingCases=negative,exactInviscidDissipationControls=inviscid,actualNestedGaussianCases=filters,zeroWidthFamilies=zero,exactTransferCounterexamples=1),indent=2))
