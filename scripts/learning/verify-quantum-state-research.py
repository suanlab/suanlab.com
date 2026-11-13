"""Independent finite-state and exact polynomial checks for the research examples."""
from fractions import Fraction as Q
import cmath,itertools,json,math
import numpy as np
inner=lambda a,b:sum(x.conjugate()*y for x,y in zip(a,b))
norm=lambda v:math.sqrt(inner(v,v).real)
normalize=lambda v:tuple(z/norm(v) for z in v)
raw=[(1+0j,0j),(0j,1+0j),(1+0j,1+0j),(1+0j,1j),(1+0j,-1j),(2+1j,3-2j),(1-2j,-2+3j)]
states=list(map(normalize,raw));triangles=nonzero=0;worst=0
for a,b,c in itertools.product(states,repeat=3):
    pairs=[inner(a,b),inner(b,c),inner(c,a)];B=math.prod(pairs);F=[abs(x)**2 for x in pairs]
    assert abs(1-sum(F)+2*B.real)<3e-14
    assert abs(abs(B)**2-math.prod(F))<3e-14
    bloch=lambda v:(2*(v[0].conjugate()*v[1]).real,2*(v[0].conjugate()*v[1]).imag,abs(v[0])**2-abs(v[1])**2)
    ra,rb,rc=map(np.array,map(bloch,[a,b,c]));candidate=(1+ra@rb+rb@rc+rc@ra+1j*(ra@np.cross(rb,rc)))/4
    assert abs(candidate-B)<3e-14
    gauged=[tuple(z*cmath.exp(1j*p) for z in v)for v,p in zip([a,b,c],[.31,-.72,1.13])]
    assert abs(inner(gauged[0],gauged[1])*inner(gauged[1],gauged[2])*inner(gauged[2],gauged[0])-B)<3e-14
    conjugated=[tuple(z.conjugate()for z in v)for v in [a,b,c]]
    assert abs(inner(conjugated[0],conjugated[1])*inner(conjugated[1],conjugated[2])*inner(conjugated[2],conjugated[0])-B.conjugate())<3e-14
    if min(map(abs,pairs))>1e-12:
        previous=a
        for v in [b,c,a]:previous=tuple(z*cmath.exp(-1j*cmath.phase(inner(previous,v)))for z in v)
        delta=max(abs(x-cmath.exp(-1j*cmath.phase(B))*y)for x,y in zip(previous,a));worst=max(worst,delta);assert delta<3e-14;nonzero+=1
    triangles+=1
# Analytic metric and finite differences, with a nonconstant gauge.
metric_checks=0
for theta,phi in itertools.product([.1,.7,1.4,2.8],[-1.2,.3,2.]):
    state=lambda s:np.array([math.cos((theta+.3*s)/2),cmath.exp(1j*(phi+.7*s))*math.sin((theta+.3*s)/2)])
    psi=state(0);d=np.array([-.15*math.sin(theta/2),cmath.exp(1j*phi)*(.15*math.cos(theta/2)+.7j*math.sin(theta/2))])
    g=np.vdot(d,d).real-abs(np.vdot(psi,d))**2
    assert abs(g-(.3**2+.7**2*math.sin(theta)**2)/4)<1e-14
    for gauge in [0.,7.]:
        dg=d+1j*gauge*psi;assert abs(np.vdot(dg,dg).real-abs(np.vdot(psi,dg))**2-g)<2e-14
        h=1e-4;assert abs((1-abs(inner(psi,state(h)*cmath.exp(1j*gauge*h)))**2)/h**2-g)<2e-6;metric_checks+=1
# Shifted Legendre recurrence, exact coefficients in ascending powers of u.
def add(a,b):
    return [(a[i] if i<len(a) else Q(0))+(b[i] if i<len(b) else Q(0)) for i in range(max(len(a),len(b)))]
def scale(a,c):return [x*c for x in a]
def mul(a,b):
    out=[Q(0)]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):out[i+j]+=x*y
    return out
integral=lambda a:sum(x/Q(i+1)for i,x in enumerate(a))
polys=[[Q(1)],[Q(-1),Q(2)]]
for n in range(1,24):polys.append(scale(add(scale(mul([Q(-1),Q(2)],polys[n]),2*n+1),scale(polys[n-1],-n)),Q(1,n+1)))
for n,p in enumerate(polys[1:],1):
    derivative=[Q(i)*p[i]for i in range(1,len(p))];assert integral(mul(p,p))==Q(1,2*n+1);assert integral(p)==0
    assert integral(mul(derivative,derivative))==2*n*(n+1)
    assert Q(2*n*(n+1)*(2*n+1),n*n)==Q(2*(n+1)*(2*n+1),n)
# Direct position-density quadrature, independently of the bound formula.
x,w=np.polynomial.legendre.leggauss(512);u=(x+1)/2;w=w/2
for n in [2,4,8,16]:
    pn=np.polynomial.legendre.legval(x,[0]*n+[1]);psi=math.sqrt(1-1/n**2)+math.sqrt(2*n+1)*pn/n;p=psi**2
    assert abs(w@p-1)<2e-12
    tv=.5*(w@np.abs(p-1));assert 0<tv<1/n
print(json.dumps({'status':'passed','triangles':triangles,'nonzeroAlignmentLoops':nonzero,'metricGaugeChecks':metric_checks,'exactPolynomialDegrees':len(polys)-1,'positionQuadratures':4,'maxAlignmentResidual':worst},indent=2))
