"""Independent symbolic generating derivatives and oscillator action transport.
No product imports. Run from the repository root with Python3 and SymPy.
"""
from pathlib import Path
import json, random
import sympy as s
rng=random.Random(2026110507)
trig=lambda e:s.trigsimp(e,method="fu")
m,F,a,b,c,lam,t,q=s.symbols('m F alpha beta c lam t q',real=True)
# Independent differentiation of the polynomial candidate, not product readouts.
S=(a+F*t)*q-a*a*t/(2*m)-a*F*t*t/(2*m)-F*F*t**3/(6*m)+c*a*a/2+lam*t*q*q/2
P=s.diff(S,q);Q=s.diff(S,a);H=P*P/(2*m)-F*q;res=s.expand(s.diff(S,t)+H)
q0=b-c*a;path=s.solve(Q-b,q)[0];v=s.diff(path,t);frozenP=P.subs(q,path)
L=m*v*v/2+F*path;integral=s.integrate(L,(t,0,t));Sinitial=S.subs({q:q0,t:0},simultaneous=True)
physical=lambda e:s.factor(e.subs(q,path))
newQ=physical(s.diff(res,a)+s.diff(res,q)*s.diff(path,a));newP=-s.diff(res,q)
M=s.Matrix([path,frozenP]).jacobian([b,a]);assert s.simplify(M.det()-1)==0
assert s.simplify(res.subs(lam,0))==0 and s.diff(S,q,a)==1
expr={'position':q,'referencePosition':path,'referenceMomentum':a+F*t,'gradientMomentum':P,'action':S,'partialTime':s.diff(S,t),'hamiltonian':H,'hamiltonJacobiResidual':res,'canonicalPosition':Q,'mixedHessian':s.diff(S,q,a),'momentumHessian':s.diff(S,a,a),'positionHessian':s.diff(S,q,q),'residualPositionDerivative':s.diff(res,q),'transformedPositionRate':s.diff(res,a)+s.diff(res,q)*s.diff(path,a),'transformedMomentumRate':newP,'frozenPositionVelocity':v,'frozenMomentum':frozenP,'frozenMomentumRate':s.diff(frozenP,t),'frozenVelocityResidual':v-frozenP/m,'frozenForceResidual':s.diff(frozenP,t)-F,'referenceEnergy':(a+F*t)**2/(2*m)-F*path,'frozenEnergy':frozenP*frozenP/(2*m)-F*path,'mechanicalAction':integral,'pathAction':S.subs(q,path),'pathActionChange':S.subs(q,path)-Sinitial,'pathActionDefect':S.subs(q,path)-Sinitial-integral}
force=[]
for i in range(240):
 p=[rng.randint(1,4),*[rng.randint(-4,4)for _ in range(5)],rng.randint(0,40),rng.randint(-4,4)]
 if i==0:p=[1,2,2,0,0,0,20,0]
 if i==1:p=[1,2,2,2,2,0,20,0]
 if i==2:p=[1,0,2,0,0,2,20,-2]
 if i%5==0:p[5]=0
 if i%11==0:p[6]=0
 vals={m:p[0],F:s.Rational(p[1],2),a:s.Rational(p[2],2),b:s.Rational(p[3],2),c:s.Rational(p[4],2),lam:s.Rational(p[5],2),t:s.Rational(p[6],20)}
 vals[q]=path.subs(vals)+s.Rational(p[7],2)
 force.append(dict(p=p,query={k:str(s.cancel(e.subs(vals)))for k,e in expr.items()},canonicalMapJacobian=[[str(s.cancel(x.subs(vals)))for x in row]for row in M.tolist()]))
# Independently reconstruct an oscillator flow and integrate its actual Lagrangian.
w,theta,x0,p0,xf,sigma,tau=s.symbols('omega theta q0 p0 xf sigma tau',real=True);C=s.cos(theta);Sn=s.sin(theta)
X=x0*C+p0*Sn/(m*w);Pflow=-m*w*x0*Sn+p0*C
Mosc=s.Matrix([X,Pflow]).jacobian([x0,p0]);assert trig(Mosc.det())==1
Dt=lambda e:s.diff(e,theta)*w
def exact_zero(e):
 u,v=s.symbols('exact_sin exact_cos',real=True)
 r=s.cancel(s.expand_trig(e.rewrite(s.sin)).subs({s.sin(theta):u,s.cos(theta):v}))
 numerator,denominator=s.fraction(r)
 # Polynomial reduction by sin²+cos²=1 is a symbolic identity check, not a tolerance.
 return s.rem(s.expand(numerator),u*u+v*v-1,u)==0

assert exact_zero(Dt(X)-Pflow/m) and exact_zero(Dt(Pflow)+m*w*w*X)
L=(Pflow*Pflow/m-m*w*w*X*X)/2
physicalAction=trig(s.integrate(s.expand_trig(L),(theta,0,theta))/w)
assert exact_zero(physicalAction-(X*Pflow-x0*p0)/2)
requiredP=s.solve(X-xf,p0)[0]
endpointAction=trig(physicalAction.subs(p0,requiredP))
endP=trig(Pflow.subs(p0,requiredP))
assert exact_zero(s.diff(endpointAction,xf)-endP)
assert exact_zero(s.diff(endpointAction,x0)+requiredP)
assert exact_zero(Dt(endpointAction)+endP**2/(2*m)+m*w*w*xf**2/2)
# Generate the mixed chart by integrating final momentum and enforcing its alpha derivative.
initialQ=s.solve(X-xf,x0)[0]
mixedP=trig(Pflow.subs(x0,initialQ))
mixed=s.integrate(mixedP,xf)-p0*p0*Sn/(2*m*w*C)
assert exact_zero(s.diff(mixed,p0)-initialQ)
assert exact_zero(Dt(mixed)+mixedP**2/(2*m)+m*w*w*xf**2/2)
A=trig(X.subs(p0,sigma*x0)/x0);Bb=trig(Pflow.subs(p0,sigma*x0)/x0)
xi=xf/A;graphAction=trig((sigma*x0*x0/2+physicalAction.subs(p0,sigma*x0)).subs(x0,xi));graphP=trig(Pflow.subs({p0:sigma*x0},simultaneous=True).subs(x0,xi))
assert exact_zero(s.diff(graphAction,xf)-graphP)
assert exact_zero(Dt(graphAction)+graphP**2/(2*m)+m*w*w*xf**2/2)
# Substituting exact rational circle values is done only after every symbolic identity.
cc,ss=s.symbols('C S',real=True)
def prepare(e):return s.cancel(s.expand_trig(e.rewrite(s.sin)).subs({s.cos(theta):cc,s.sin(theta):ss}))
def value(e,vals):return str(s.cancel(prepare(e).subs(vals)))
baseExpr={'position':X,'momentum':Pflow,'energy':Pflow**2/(2*m)+m*w*w*X**2/2,'physicalAction':physicalAction,'endpointSensitivity':s.diff(X,p0),'mixedSensitivity':s.diff(X,x0),'graphProjection':A,'graphMomentumCoefficient':Bb}
osc=[]
for i in range(280):
 p=[rng.randint(1,4),rng.randint(1,4),rng.randint(-4,4),rng.randint(-4,4),rng.randint(0,3),rng.randint(0,4),rng.randint(-4,4),rng.randint(-4,4)]
 presets=[[1,1,2,2,1,0,2,0],[1,1,2,2,2,0,-2,0],[1,1,2,0,0,2,0,-3],[1,1,0,2,1,0,2,2],[1,1,2,2,0,0,2,0]]
 if i<len(presets):p=presets[i]
 u=s.Rational(p[5],4);cos=(1-u*u)/(1+u*u);sin=2*u/(1+u*u);cos,sin=[(cos,sin),(-sin,cos),(-cos,-sin),(sin,-cos)][p[4]]
 vals={m:p[0],w:p[1],x0:s.Rational(p[2],2),p0:s.Rational(p[3],2),xf:s.Rational(p[6],2),sigma:s.Rational(p[7],4),cc:cos,ss:sin}
 query={k:value(e,vals)for k,e in baseExpr.items()};query.update(cosine=str(cos),sine=str(sin))
 query['endpointChart']=None if sin==0 else {k:value(e,vals)for k,e in {'action':endpointAction,'initialMomentum':requiredP,'finalMomentum':endP,'partialTime':Dt(endpointAction),'mixedHessian':s.diff(endpointAction,xf,x0)}.items()}
 query['mixedChart']=None if cos==0 else {k:value(e,vals)for k,e in {'action':mixed,'initialPosition':initialQ,'finalMomentum':mixedP,'partialTime':Dt(mixed),'mixedHessian':s.diff(mixed,xf,p0)}.items()}
 av=s.sympify(query['graphProjection']);query['graphChart']=None if av==0 else{k:value(e,vals)for k,e in {'action':graphAction,'momentum':graphP,'hessian':s.diff(graphAction,xf,xf),'initialLabel':xi,'partialTime':Dt(graphAction)}.items()}
 osc.append(dict(p=p,query=query,flowMatrix=[[value(e,vals)for e in row]for row in Mosc.tolist()]))
Path('scripts/fixtures/hamilton-jacobi-independent.json').write_text(json.dumps(dict(method='Independent SymPy generating derivatives,canonical Jacobians,integrated oscillator Lagrangian and initial-action transport; symbolic PDE/endpoint identities before rational-clock substitution; no product imports',uniformForce=force,oscillator=osc),indent=2)+'\n')
print('Generated',len(force),'polynomial HJ and',len(osc),'oscillator action/chart references; symbolic canonical,endpoint,PDE and transported-action identities passed.')
