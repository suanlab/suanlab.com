"""Independent symbolic checks for the analytic manuscript examples,not a continuum solver."""
import json
from pathlib import Path
import sympy as s
x=s.Symbol('x',real=True);alpha,m,omega,hbar=s.symbols('alpha m omega hbar',positive=True)
psi=(alpha/s.pi)**s.Rational(1,4)*s.exp(-alpha*x*x/2)
Hpsi=s.simplify(-hbar*hbar*s.diff(psi,x,2)/(2*m)+m*omega*omega*x*x*psi/2)
mean=s.simplify(s.integrate(psi*Hpsi,(x,-s.oo,s.oo)))
variance=s.simplify(s.integrate(Hpsi**2,(x,-s.oo,s.oo))-mean**2)
assert s.simplify(mean-hbar*hbar*alpha/(4*m)-m*omega*omega/(4*alpha))==0
assert s.simplify(variance-(m*omega*omega-hbar*hbar*alpha*alpha/m)**2/(8*alpha*alpha))==0
astar=m*omega/hbar
annihilation=s.sqrt(m*omega/(2*hbar))*x*psi+s.sqrt(hbar/(2*m*omega))*s.diff(psi,x)
assert s.simplify(annihilation.subs(alpha,astar))==0
f=s.Function('f')(x);a=lambda g:s.sqrt(m*omega/(2*hbar))*x*g+s.sqrt(hbar/(2*m*omega))*s.diff(g,x)
ad=lambda g:s.sqrt(m*omega/(2*hbar))*x*g-s.sqrt(hbar/(2*m*omega))*s.diff(g,x)
assert s.simplify(hbar*omega*(ad(a(f))+f/2)-(-hbar*hbar*s.diff(f,x,2)/(2*m)+m*omega*omega*x*x*f/2))==0
q=s.Symbol('q',real=True);S=1+q*q;A=2*q*q;E=A/S;derivative=s.simplify(s.diff(E,q));metric=s.simplify((s.diff(A,q)-E*s.diff(S,q))/S);wrong=s.simplify(s.diff(A,q)/S)
assert s.simplify(derivative-metric)==0
u=s.Symbol('u',real=True);path=(1-2*u+u*u)/(1+u*u)
mu=(4-s.sqrt(7))/3;ratio=s.simplify((2*mu-3)/(3-mu));B=s.Matrix([[1,0],[0,1],[1,1]]);H=s.diag(0,1,3);gram=B.H*B;energy=B.H*H*B;z=B*s.Matrix([1,ratio]);res=s.simplify((H-mu*s.eye(3))*z)
assert res==mu*s.Matrix([-1,-1,1]);assert s.simplify(B.H*res)==s.zeros(2,1)
M=s.Matrix([[2,1+s.I],[0,1]]);gs=s.simplify(M.H*gram*M);ga=s.simplify(M.H*energy*M)
t=s.Symbol('t');assert s.expand((t*gs-ga).det()-4*(3*t*t-8*t+3))==0
payload=dict(method='Independent SymPy exact differentiation,integration and matrix identities',continuumEngineImplemented=False,gaussian=dict(energy=str(mean),variance=str(variance),normalization=str(s.simplify(s.integrate(psi**2,(x,-s.oo,s.oo)))),stationaryAlpha=str(astar),groundEnergy=str(s.simplify(mean.subs(alpha,astar))),doubleAlphaEnergyInHbarOmega=str(s.simplify(mean.subs(alpha,2*astar)/(hbar*omega))),doubleAlphaVarianceInHbarOmegaSquared=str(s.simplify(variance.subs(alpha,2*astar)/(hbar*omega)**2)),annihilationIdentity=True),movingMetric=dict(correctDerivative=str(derivative),atOne=str(derivative.subs(q,1)),withoutOverlapDerivativeAtOne=str(wrong.subs(q,1))),directionalPath=dict(firstAtZero=str(s.diff(path,u).subs(u,0)),secondAtZero=str(s.diff(path,u,2).subs(u,0)),energyAtOne=str(path.subs(u,1))),ritz=dict(minimum=str(mu),coefficientRatio=str(ratio),fullResidualDirection=[-1,-1,1],gramAfterChange=[[str(gs[i,j])for j in range(2)]for i in range(2)],hamiltonianAfterChange=[[str(ga[i,j])for j in range(2)]for i in range(2)],pencilFactor=4))
path=Path('scripts/fixtures/quantum-variational-authoring-reference.json');path.write_text(json.dumps(payload,indent=2)+'\n');print('Analytic Gaussian,metric derivative,directional path and Ritz manuscript identities verified.')
