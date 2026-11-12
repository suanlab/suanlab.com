"""Independent finite checks of the stated modal and loaded-leg research models.

Standard-library algebra, a Taylor propagator, RK4 time quadrature and spatial
Simpson quadrature supplement the analytic arguments, not a general solver.
"""
from fractions import Fraction as F
from math import sqrt,sin,cos,isclose
import json

def mv(M,x):return [sum(a*b for a,b in zip(row,x)) for row in M]
def matrix(w,g,d):return [[0,w,0],[-w,0,g],[0,-g,-d]]
def norm2(x):return sum(v*v for v in x)
def prop(M,x,t):
 term=list(x);out=list(x)
 for n in range(1,120):
  term=[v*t/n for v in mv(M,term)];out=[a+b for a,b in zip(out,term)]
  if max(map(abs,term))<1e-18:break
 return out

def determinant(M):
 a,b,c=M[0];d,e,f=M[1];g,h,i=M[2]
 return a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)
routh=0
for w in [F(1,2),F(1),F(2)]:
 for g in [F(-1),F(1,4),F(2)]:
  for d in [F(1,100),F(1,10),F(1),F(10),F(100)]:
   M=matrix(w,g,d)
   # Three independently computed characteristic coefficients.
   a=-sum(M[i][i] for i in range(3))
   b=sum(M[i][i]*M[j][j]-M[i][j]*M[j][i] for i in range(3) for j in range(i+1,3))
   c=-determinant(M)
   assert (a,b,c)==(d,w*w+g*g,d*w*w)
   assert a>0 and b>0 and c>0 and a*b-c==d*g*g>0
   routh+=1
adiabatic=0
for w in [.5,1.,2.]:
 for g in [-1.,.25,2.]:
  M=matrix(w,g,0.);n=[g,0.,w];assert max(map(abs,mv(M,n)))==0
  x=[.2,-.1,.3];Om=sqrt(w*w+g*g);Mx=mv(M,x);MMx=mv(M,Mx)
  for t in [.1,1.,2.]:
   closed=[a+sin(Om*t)/Om*b+(1-cos(Om*t))/(Om*Om)*c for a,b,c in zip(x,Mx,MMx)]
   actual=prop(M,x,t)
   assert max(abs(a-b) for a,b in zip(actual,closed))<2e-12
   assert isclose(norm2(actual),norm2(x),abs_tol=2e-12)
   assert isclose(sum(a*b for a,b in zip(n,actual)),sum(a*b for a,b in zip(n,x)),abs_tol=2e-12)
   adiabatic+=1
rk=0
for w in [.5,1.,2.]:
 for g in [-1.,.25,2.]:
  for d in [.1,1.,10.]:
   M=matrix(w,g,d);x0=[.2,-.1,.3];state=x0+[0.];t=.5;n=1000;h=t/n
   def deriv(y):return mv(M,y[:3])+[2*d*y[2]*y[2]]
   for _ in range(n):
    k1=deriv(state);k2=deriv([a+h*b/2 for a,b in zip(state,k1)])
    k3=deriv([a+h*b/2 for a,b in zip(state,k2)]);k4=deriv([a+h*b for a,b in zip(state,k3)])
    state=[a+h*(b+2*c+2*e+f)/6 for a,b,c,e,f in zip(state,k1,k2,k3,k4)]
   assert max(abs(a-b) for a,b in zip(state[:3],prop(M,x0,t)))<2e-11
   assert state[3]>0 and abs(norm2(state[:3])+state[3]-norm2(x0))<2e-11
   rk+=1
# Exact zero conduction and zero coupling are distinct persistent controls.
zero=0
for w in [F(1,2),F(1),F(2)]:
 M=matrix(w,F(1),F(0));n=[F(1),F(0),w];assert mv(M,n)==[0,0,0];zero+=1
 M=matrix(w,F(0),F(2));assert M[0][2]==M[1][2]==M[2][0]==M[2][1]==0;zero+=1

exact=0
for tc,th in [(F(300),F(310)),(F(300),F(400)),(F(250),F(500))]:
 for R,H,S in [(F(1,10),F(1,5),F(3,100)),(F(1),F(1),F(1,100)),(F(1,100),F(1,10),F(1,20))]:
  for m in [F(1,2),F(1),F(2),F(4),F(10)]:
   dt=th-tc;I=S*dt/(R*(1+m));Qh=S*th*I+H*dt-R*I*I/2;Qc=S*tc*I+H*dt+R*I*I/2;P=m*R*I*I
   assert Qh>0 and Qc>0 and Qh-Qc==P==S*dt*I-R*I*I
   assert Qc/tc-Qh/th==H*dt*dt/(tc*th)+R*I*I*(th+tc)/(2*tc*th)>0
   # Recover the original bar fields independently, L=A=1, j=-I.
   j=-I;Tx0=dt+R*j*j/(2*H);Tx1=dt-R*j*j/(2*H)
   assert -(S*tc*j-H*Tx0)==Qc
   assert -(S*th*j-H*Tx1)==Qh
   assert -j*(R*j+S*dt)==P
   exact+=1
entropy=0;optimum=0
for tc,th in [(300.,310.),(300.,400.),(250.,500.)]:
 for R,H,S in [(.1,.2,.03),(1.,1.,.01),(.01,.1,.05)]:
  dt=th-tc;Z=S*S/(R*H);mo=sqrt(1+Z*(tc+th)/2)
  def obs(m):
   I=S*dt/(R*(1+m));Qh=S*th*I+H*dt-R*I*I/2;Qc=S*tc*I+H*dt+R*I*I/2;P=I*I*m*R
   return I,Qh,Qc,P,P/Qh
  for m in [.1,1.,mo,10.,100.]:
   I,Qh,Qc,P,eta=obs(m);j=-I;n=40000
   def density(x):
    T=tc+dt*x+R*j*j/(2*H)*x*(1-x);Tx=dt+R*j*j/(2*H)*(1-2*x)
    return R*j*j/T+H*Tx*Tx/(T*T)
   quadrature=(density(0)+density(1)+4*sum(density(i/n) for i in range(1,n,2))+2*sum(density(i/n) for i in range(2,n,2)))/(3*n)
   assert isclose(quadrature,Qc/tc-Qh/th,rel_tol=2e-9,abs_tol=1e-12)
   assert 0<eta<dt/th;entropy+=1
  assert obs(1.)[3]>=obs(mo)[3] and obs(mo)[-1]>=obs(1.)[-1]
  eps=1e-5;assert abs((obs(mo*(1+eps))[-1]-obs(mo*(1-eps))[-1])/(2*eps))<1e-9
  assert isclose(obs(mo)[-1],dt/th*(mo-1)/(mo+tc/th),rel_tol=1e-12);optimum+=1
print(json.dumps(dict(exactCubicAndHurwitzCases=routh,adiabaticPropagatorCases=adiabatic,independentTimeQuadratureCases=rk,exactPersistentZeroControls=zero,exactLoadedLegPortCases=exact,localEntropyQuadratureCases=entropy,distinctObjectiveOptimumCases=optimum),indent=2))
