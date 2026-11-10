"""Independent direct elliptic cn/sn/dn reference; production uses nome Fourier.
Numerical 200-digit comparisons and SymPy polynomial identities, not interval proofs.
"""
from pathlib import Path
from fractions import Fraction as F
import json,random
import mpmath as mp
import sympy as sp
mp.mp.dps=200
S=2**384
cases=json.loads(Path('scripts/fixtures/duffing-secular-examples.json').read_text())
rng=random.Random(20261110)
for i in range(14):
    cases.append({'id':f'seeded-{i:02}', 'input':{'lambda':str(F(rng.randint(1,16),rng.randint(1,128))),'amplitude':str(F(rng.randint(1,12),rng.randint(1,9))),'horizon':str(F(rng.randint(1,10000),rng.randint(1,13))),'probeFraction':str(F(rng.randint(0,16),16))}})
for amplitude in ['1/999999999989','16']:
    cases.append({'id':f'extreme-{amplitude.replace("/","-")}', 'input':{'lambda':'1/999999999983','amplitude':amplitude,'horizon':'999999999979','probeFraction':'1'}})
def m(v):
    v=F(v);return mp.mpf(v.numerator)/v.denominator
def cell(v):return str(int(mp.floor(v*S)))
z,G=sp.symbols('z G')
w=[1+sp.Rational(3,8)*G,1+sp.Rational(3,8)*G-sp.Rational(21,256)*G**2]
shapes=[(1-G/32)*sp.chebyshevt(1,z)+G/32*sp.chebyshevt(3,z),(1-G/32+sp.Rational(23,1024)*G**2)*sp.chebyshevt(1,z)+(G/32-sp.Rational(24,1024)*G**2)*sp.chebyshevt(3,z)+G**2/1024*sp.chebyshevt(5,z)]
residuals=[]
for i,(freq,shape) in enumerate(zip(w,shapes)):
    # d²/dτ² P(cosτ)=(1-z²)P″(z)-zP′(z), independently of harmonic code.
    residual=sp.Poly(sp.expand(freq**2*((1-z*z)*sp.diff(shape,z,2)-z*sp.diff(shape,z))+shape+G*shape**3),z)
    assert all(sp.expand(c).coeff(G,j)==0 for c in residual.all_coeffs() for j in range(i+2))
    residuals.append(residual)
rows=[]
for c in cases:
    v=c['input'];lam,A,H,P=[F(v[k]) for k in ['lambda','amplitude','horizon','probeFraction']];g=lam*A*A
    aa,gg,hh=map(m,[A,g,H]);mparam=gg/(2*(1+gg))
    freq=mp.pi*mp.sqrt(1+gg)/(2*mp.ellipk(mparam))
    freqs=[1,1+mp.mpf(3)*gg/8,1+mp.mpf(3)*gg/8-mp.mpf(21)*gg**2/256]
    def reference(t):
        tt=m(t)
        if not A:return(mp.mpf(0),mp.mpf(0))
        if not t:return(aa,mp.mpf(0))
        u=mp.sqrt(1+gg)*tt
        return(aa*mp.ellipfun('cn',u,mparam),-aa*mp.sqrt(1+gg)*mp.ellipfun('sn',u,mparam)*mp.ellipfun('dn',u,mparam))
    def naive(t):return aa*(mp.cos(t)+gg*((mp.cos(3*t)-mp.cos(t))/32-mp.mpf(3)*t*mp.sin(t)/8))
    def first(t):
        tau=freqs[1]*t
        return aa*(mp.cos(tau)+gg*(mp.cos(3*tau)-mp.cos(tau))/32)
    def second(t):
        tau=freqs[2]*t
        return aa*(mp.cos(tau)+gg*(mp.cos(3*tau)-mp.cos(tau))/32+gg**2*(23*mp.cos(tau)-24*mp.cos(3*tau)+mp.cos(5*tau))/1024)
    functions=[naive,first,second]
    def sample(t):
        pos,vel=reference(t);return{'time':str(t),'exactPosition':cell(pos),'exactVelocity':cell(vel),**dict(zip(['naivePosition','firstPosition','secondPosition'],[cell(fn(m(t))) for fn in functions]))}
    probeTime=H*P;pos,vel=reference(probeTime);methods=[]
    E=A*A/2+lam*A**4/4
    for fn in functions:
        t=m(probeTime);p=fn(t);vv=mp.diff(fn,t);acc=mp.diff(fn,t,2);energy=vv*vv/2+p*p/2+m(lam)*p**4/4
        methods.append({'position':cell(p),'velocity':cell(vv),'residual':cell(acc+p+m(lam)*p**3),'energy':cell(energy),'energyDefect':cell(energy-m(E)),'positionError':cell(abs(p-pos)),'velocityError':cell(abs(vv-vel))})
    width=min(H,F(32,int(mp.ceil(freq))));start=max(F(0),min(probeTime-width/2,H-width))
    coeffs=[]
    for poly in residuals:
        p=sp.Poly(poly.as_expr().subs(G,sp.Rational(g.numerator,g.denominator)),z);co=[F(p.nth(j)) for j in range(16)];coeffs.append({'normalizedResidualPolynomialInCos':[str(x) for x in co],'normalizedResidualUniformBound':str(sum(map(abs,co),F(0))),'physicalResidualUniformBound':str(A*sum(map(abs,co),F(0)))})
    rows.append({**c,'effectiveCoupling':str(g),'initialEnergy':str(E),'frequency':None if not A else cell(freq),'period':None if not A else cell(2*mp.pi/freq),'phaseDrift':None if not A else [cell((nu-freq)*hh) for nu in freqs],'probe':{**sample(probeTime),'methods':methods},'fullSamples':[sample(H*F(i,64)) for i in range(65)],'localSamples':[sample(start+width*F(i,64)) for i in range(65)],'localWindow':{'start':str(start),'width':str(width)},'residuals':coeffs})
    print(c['id'],flush=True)
Path('scripts/fixtures/duffing-secular-reference.json').write_text(json.dumps({'decimalPrecision':200,'referenceQuantumBits':384,'symbolicResidualLowestPowers':[2,3],'formula':'Direct mpmath elliptic cn/sn/dn and SymPy cos-polynomial operator; numerical cells are not formal enclosures.','cases':rows},indent=2)+'\n')
print(json.dumps({'referenceCases':len(rows),'physicalSamples':65*len(rows),'localSamples':65*len(rows)}))
