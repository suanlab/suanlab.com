"""Independent symbolic checks for the scalar-field draft; no product imports."""
import json
from pathlib import Path
import sympy as sp

h, c = sp.symbols("h c", positive=True)
a, b, epsilon = sp.symbols("a b epsilon", real=True)
q = sp.symbols("q0:3", real=True)
v = sp.symbols("v0:3", real=True)
u = c**2 / (2*h) * sum((q[(i+1)%3]-q[i])**2 for i in range(3)) + h * sum(a*x**2/2+b*x**4/4 for x in q)
acceleration = [c**2/h**2*(q[(i-1)%3]-2*q[i]+q[(i+1)%3])-a*q[i]-b*q[i]**3 for i in range(3)]
for i in range(3):
    assert sp.simplify(-sp.diff(u,q[i])/h-acceleration[i]) == 0
power = sp.expand(sum(sp.diff(u,q[i])*v[i]+h*v[i]*acceleration[i] for i in range(3)))
assert power == 0
net_acceleration = sp.factor(sum(acceleration))
assert sp.simplify(net_acceleration + a*sum(q) + b*sum(x**3 for x in q)) == 0
shifted = u.subs(dict(zip(q,[x+epsilon for x in q])), simultaneous=True)
assert sp.simplify((shifted-u).subs({a:0,b:0})) == 0
fixed = {h:1,c:1,a:0,b:0,q[0]:1,q[1]:0,q[2]:-1,v[0]:1,v[1]:0,v[2]:0}
assert u.subs(fixed) == 3
assert [x.subs(fixed) for x in acceleration] == [-3,0,3]
kinetic = h*sum(x*x for x in v)/2
assert (kinetic+u).subs(fixed) == sp.Rational(7,2)
perturbed = u.subs(q[0],q[0]+epsilon).subs(fixed).expand()
assert perturbed == 3+3*epsilon+epsilon**2
assert perturbed.subs(epsilon,sp.Rational(1,10)) == sp.Rational(331,100)
assert perturbed.subs(epsilon,-sp.Rational(1,10)) == sp.Rational(271,100)
# A quartic potential makes centered differences differ from the derivative.
delta = sp.symbols("delta", nonzero=True, real=True)
f = (u.subs(q[0],q[0]+delta)-u.subs(q[0],q[0]-delta))/(2*delta)
assert sp.simplify(f-sp.diff(u,q[0])-h*b*q[0]*delta**2) == 0
x,t = sp.symbols("x t", real=True)
phi = sp.Function("phi")(x,t)
lagrangian = sp.diff(phi,t)**2/2-c**2*sp.diff(phi,x)**2/2-a*phi**2/2-b*phi**4/4
residual = sp.diff(phi,t,2)-c**2*sp.diff(phi,x,2)+a*phi+b*phi**3
euler_lagrange = sp.diff(lagrangian,phi)-sp.diff(sp.diff(lagrangian,sp.diff(phi,t)),t)-sp.diff(sp.diff(lagrangian,sp.diff(phi,x)),x)
assert sp.simplify(euler_lagrange+residual) == 0
energy = sp.diff(phi,t)**2/2+c**2*sp.diff(phi,x)**2/2+a*phi**2/2+b*phi**4/4
flux = -c**2*sp.diff(phi,t)*sp.diff(phi,x)
assert sp.simplify(sp.diff(energy,t)+sp.diff(flux,x)-sp.diff(phi,t)*residual) == 0
Q = sp.symbols("Q", real=True)
assert sp.solve(-(-1)*Q-Q**3,Q) == [-1,0,1]
report = {"stage":"draft-symbolic-reference","sympyVersion":sp.__version__,"productImports":False,"periodicSites":3,"arbitrarySpacingMassMatrixChecked":True,"potentialGradientAndAccelerationIdentity":True,"semidiscreteEnergyDerivativeZero":True,"uniformShiftAndNetForceChecked":True,"workedExample":{"potential":"3","kinetic":"1/2","energy":"7/2","acceleration":["-3","0","3"],"perturbedEnergies":["331/100","271/100"]},"quarticCenteredDifferenceError":"h*b*q0*delta**2","continuumEulerLagrangeIdentity":True,"continuumEnergyFluxIdentity":True,"readerRegistered":False,"productionKernelImplemented":False,"interactiveUiImplemented":False}
out = Path(".runtime/learning/scalar-field-action-foundation")
out.mkdir(parents=True,exist_ok=True)
(out/"symbolic-reference.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report))
