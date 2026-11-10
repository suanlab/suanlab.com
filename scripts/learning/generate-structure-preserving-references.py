"""Independent symbolic map derivatives,DOP853 trajectories and80-digit reduced-circle flows."""
from pathlib import Path
import json,random
import sympy as s
import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp
mp.mp.dps=80;rng=random.Random(2026110511)
q,p,h,m,k,a=s.symbols('q p h m k a',real=True)
U=lambda x:k*x*x/2+a*x**4/4
G=lambda x:s.diff(U(q),q).subs(q,x)
H=lambda x,y:y*y/(2*m)+U(x)
ph=p-h*G(q)/2;Q=q+h*ph/m;P=ph-h*G(Q)/2
J=s.Matrix([Q,P]).jacobian([q,p]);assert s.factor(J.det()-1)==0
Ld=m*(s.Symbol('Q')-q)**2/(2*h)-h*(U(q)+U(s.Symbol('Q')))/2
assert s.factor((-s.diff(Ld,q)).subs(s.Symbol('Q'),Q)-p)==0
assert s.factor(s.diff(Ld,s.Symbol('Q')).subs(s.Symbol('Q'),Q)-P)==0
shadow=lambda x,y:H(x,y)+h*h*((k+3*a*x*x)*y*y/(12*m*m)-G(x)**2/(24*m))
assert s.simplify(s.series(shadow(Q,P)-shadow(q,p),h,0,5).removeO())==0
rhs=lambda x,y:s.Matrix([y/m,-G(x)]);z=s.Matrix([q,p]);d1=rhs(q,p);d2=rhs(*(z+h*d1/2));d3=rhs(*(z+h*d2/2));d4=rhs(*(z+h*d3));RK=z+h*(d1+2*d2+2*d3+d4)/6
W=H(q,p)-U(Q)
arg=(q,p,h,m,k,a)
funcs={name:s.lambdify(arg,list(expr)+list(s.Matrix(expr).jacobian([q,p])),'mpmath',cse=True)for name,expr in [('verlet',[Q,P]),('rk4',RK)]}
qfun=s.lambdify(arg,Q,'mpmath',cse=True);pfun=s.lambdify(arg,P,'mpmath',cse=True);wfun=s.lambdify(arg,W,'mpmath',cse=True)
steps=[]
for i in range(120):
 mass=rng.randint(1,4);stiff=rng.randint(0,4);quartic=rng.randint(0,4);qi=s.Rational(rng.randint(-4,4),2);pi=s.Rational(rng.randint(-4,4),2);step=s.Rational(rng.randint(0,8),16)
 if i==0:mass,stiff,quartic,qi,pi,step=1,1,1,s.Rational(1),s.Rational(0),s.Rational(1,8)
 if i==1:mass,stiff,quartic,qi,pi,step=1,4,4,s.Rational(2),s.Rational(2),s.Rational(1)
 if i==2:mass,stiff,quartic,qi,pi,step=1,4,0,s.Rational(1),s.Rational(0),s.Rational(1)
 if i==3:mass,stiff,quartic,qi,pi,step=1,0,0,s.Rational(0),s.Rational(0),s.Rational(1,2)
 vals={q:qi,p:pi,h:step,m:mass,k:stiff,a:quartic};args=[mp.mpf(str(v))if not isinstance(v,s.Rational)else mp.mpf(int(v.p))/int(v.q)for v in [qi,pi,step,mass,stiff,quartic]]
 out={}
 for name,fn in funcs.items():
  raw=fn(*args);out[name]={'state':[float(v)for v in raw[:2]],'jacobian':[[float(raw[2]),float(raw[3])],[float(raw[4]),float(raw[5])]]}
 rationalW=s.cancel(W.subs(vals));trialP=s.cancel(P.subs(vals));targetQ=qfun(*args)
 if step==0 or stiff==quartic==0:out['energy-projected']=dict(out['verlet'],status='regular')
 elif rationalW<0:out['energy-projected']={'state':None,'jacobian':None,'status':'energy-target-unreachable'}
 elif rationalW==0:out['energy-projected']={'state':[float(targetQ),0.],'jacobian':None,'status':'projection-boundary'}
 elif trialP==0:out['energy-projected']={'state':None,'jacobian':None,'status':'projection-sign-undetermined'}
 else:
  pp=mp.sign(pfun(*args))*mp.sqrt(2*mass*wfun(*args));A=mp.mpf(out['verlet']['jacobian'][0][0]);Bb=mp.mpf(step.p)/int(step.q)/mass
  # Differentiate the exact Q with SymPy,not rounded matrix entries.
  A=s.diff(Q,q).subs(vals);A=mp.mpf(str(s.N(A,82)));grad=lambda x:stiff*x+quartic*x**3
  pq=mass/pp*(grad(args[0])-grad(targetQ)*A);ppd=mass/pp*(args[1]/mass-grad(targetQ)*Bb)
  out['energy-projected']={'state':[float(targetQ),float(pp)],'jacobian':[[float(A),float(Bb)],[float(pq),float(ppd)]],'status':'regular'}
 steps.append(dict(parameters=dict(mass=mass,stiffness=stiff,quartic=quartic),position=float(qi),momentum=float(pi),stepSize=float(step),methods=out))
trajectories=[]
for i in range(24):
 controls=[rng.randint(1,4),rng.randint(0,4),rng.randint(0,4),rng.randint(-4,4),rng.randint(-4,4),rng.randint(1,8),rng.randint(0,4),0]
 if i==0:controls=[1,1,1,2,0,4,2,16]
 if i==1:controls=[1,4,4,4,4,8,0,4]
 mass,stiff,quartic=controls[:3];T=controls[5]/2;N=4*2**controls[6];times=np.linspace(0,T,N+1);y0=[controls[3]/2,controls[4]/2]
 def f(t,z):return[z[1]/mass,-stiff*z[0]-quartic*z[0]**3]
 sol=solve_ivp(f,[0,T],y0,method='DOP853',rtol=2.5e-14,atol=2e-14,t_eval=times)
 check=solve_ivp(f,[0,T],y0,method='DOP853',rtol=2e-13,atol=2e-13,t_eval=times)
 assert sol.success and check.success and np.max(np.abs(sol.y-check.y))<2e-10
 trajectories.append(dict(p=controls,states=[dict(time=float(t),position=float(qv),momentum=float(pv))for t,qv,pv in zip(times,*sol.y)],toleranceRefinementDifference=float(np.max(np.abs(sol.y-check.y)))))
# Symbolically derive the reduced normalized-drift projection and its area contraction.
l,I,hh=s.symbols('ell I h',real=True,positive=True);v=hh*l/I;Lp=l/s.sqrt(1+v*v)
assert s.simplify(s.diff(Lp,l)-(1+v*v)**s.Rational(-3,2))==0
circles=[]
for i in range(120):
 controls=[rng.randint(1,4),rng.randint(1,4),rng.randint(-4,4),rng.randint(0,40),rng.randint(1,32),rng.randint(0,3),rng.randint(0,4),0]
 presets=[[1,2,2,10,8,0,0,8],[1,2,2,20,4,0,0,1],[1,2,2,22,4,0,0,1],[1,2,0,10,8,2,2,8],[1,2,-2,10,8,0,2,8],[2,3,2,0,8,1,3,8]]
 if i<len(presets):controls=presets[i]
 mass=mp.mpf(controls[0]);R=mp.mpf(controls[1])/2;pt=mp.mpf(controls[2])/2;step=mp.mpf(controls[3])/20;N=controls[4];u=mp.mpf(controls[6])/4;theta=controls[5]*mp.pi/2+2*mp.atan(u);L=R*pt;inertia=mass*R*R;nu=step*L/inertia;disc=1-nu*nu;classification='regular'if abs(controls[3]*controls[2])<20*controls[0]*controls[1]else'boundary'if abs(controls[3]*controls[2])==20*controls[0]*controls[1]else'unreachable'
 def phase_of(Lv,n):return sum(mp.atan(step*Lv/inertia/mp.sqrt(1+j*(step*Lv/inertia)**2))for j in range(n))
 runs={}
 for name in ['rattle','projected']:
  rows=[]
  for n in range((1 if classification=='unreachable'and name=='rattle'else N+1)):
   phase=n*mp.asin(nu)if name=='rattle'and n else phase_of(L,n)if name=='projected'else mp.mpf(0)
   currentL=L if name=='rattle'else L/mp.sqrt(1+n*nu*nu);angle=theta+phase
   xy=[R*mp.cos(angle),R*mp.sin(angle)];pp=[-currentL/R*mp.sin(angle),currentL/R*mp.cos(angle)]
   if n==0:jac=[[1.,0.],[0.,1.]]
   elif name=='rattle':jac=[[1.,float(n*step/(inertia*mp.sqrt(disc)))],[0.,1.]]if classification=='regular'else None
   else:jac=[[1.,float(mp.diff(lambda LL:phase_of(LL,n),L))],[0.,float(mp.diff(lambda LL:LL/mp.sqrt(1+n*(step*LL/inertia)**2),L))]]
   rows.append(dict(position=list(map(float,xy)),momentum=list(map(float,pp)),phaseAdvance=float(phase),phaseError=float(phase-n*nu),energy=float(currentL*currentL/(2*inertia)),angularMomentum=float(currentL),jacobian=jac))
  runs[name]=rows
 circles.append(dict(p=controls,admissibility=classification,runs=runs))
Path('scripts/fixtures/structure-preserving-independent.json').write_text(json.dumps(dict(method='Independent SymPy polynomial map differentiation/discrete Legendre/modified-action series;DOP853 tolerance-refined physical trajectories;80digit reduced-circle phase and derivative references;no product imports',quarticSteps=steps,quarticTrajectories=trajectories,circles=circles),indent=2)+'\n')
print('Generated120quartic step systems(360method queries),24DOP853 trajectories and120circle runs;symbolic variational,shadow-series and reduced area identities passed.')
