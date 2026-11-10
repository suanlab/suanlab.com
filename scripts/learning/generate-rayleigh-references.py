"""Independent SymPy exact and mpmath Hermitian spectral references (250 dps)."""
import json
import random
from pathlib import Path
import sympy as s
import mpmath as mp

mp.mp.dps = 250
OUT = Path('scripts/fixtures/rayleigh-reference.json')
CASES = []

def preparation(diagonal, entries=None, state=None, direction=None, denominator=1, path=(1, 3)):
    d = len(diagonal)
    p = dict(dimension=d, hDenominator=denominator)
    entries = entries or {}
    for i in range(4):
        for j in range(i, 4):
            v = diagonal[i] if i == j and i < d else entries.get((i, j), 0)
            re, im = v if isinstance(v, tuple) else (v, 0)
            p[f'hReal{i}{j}'] = re
            if i != j:
                p[f'hImag{i}{j}'] = im
    for prefix, vector in [('state', state or [1] + [0]*(d-1)), ('direction', direction if direction is not None else [0]*d)]:
        for i in range(4):
            v = vector[i] if i < d else 0
            re, im = v if isinstance(v, tuple) else (v, 0)
            p[f'{prefix}Real{i}'], p[f'{prefix}Imag{i}'] = re, im
    p['pathNumerator'], p['pathDenominator'] = path
    assert len(p) == 36
    return p

def add(label, **kwargs):
    CASES.append((label, preparation(**kwargs)))

add('scalar negative', diagonal=[-3], state=[(2, -1)], direction=[(1, 2)])
add('scalar zero', diagonal=[0])
add('ground eigenstate', diagonal=[0, 2])
add('excited eigenstate', diagonal=[0, 2], state=[0, 1])
add('equal phase superposition', diagonal=[0, 2], state=[1, (0, 1)], direction=[(2, 1), -1])
add('zero path state', diagonal=[0, 2], state=[1, (0, 1)], direction=[-1, (0, -1)], path=(1, 1))
add('degenerate ground plane', diagonal=[0, 0, 3], state=[1, (0, 2), 1], direction=[1, -1, (0, 1)])
add('ground vector in degenerate plane', diagonal=[0, 0, 3], state=[1, (0, 2), 0])
add('complex degenerate ground', diagonal=[1, 1, 1], entries={(0, 1):(0, -1), (0, 2):1, (1, 2):(0, 1)}, denominator=3, state=[1, 2, (1, -1)], direction=[1, (0, 1), -1])
add('constant four dimensional', diagonal=[-7]*4, state=[1, (1, 2), -3, (2, -1)], direction=[-1, 1, (0, 1), 2])
add('tiny positive variance', diagonal=[0, 1], denominator=1000000, state=[1000000, 1], direction=[1, -1])
add('negative near zero ground', diagonal=[1000000, 0], entries={(0, 1):1}, denominator=1000000, state=[1, (1, 1)])
add('rational non dyadic spectrum', diagonal=[-1, 2, 2], denominator=3, state=[1, 1, (0, 1)])
add('zero direction', diagonal=[-2, 1, 4, 7], state=[1, 2, 3, 4])
add('four distinct signed roots', diagonal=[-5, -1, 2, 8], entries={(0, 1):(1, 2), (0, 3):-2, (1, 2):(0, 3), (2, 3):2}, state=[(1, 2), -2, (3, -1), 1], direction=[2, (0, -1), 1, -3])
rng = random.Random(140811)
for d in range(1, 5):
    for k in range(7):
        diagonal = [rng.randint(-9, 9) for _ in range(d)]
        entries = {(i, j):(rng.randint(-4, 4), rng.randint(-4, 4)) for i in range(d) for j in range(i+1, d)}
        state = [(rng.randint(-4, 4), rng.randint(-4, 4)) for _ in range(d)]
        if all(v == (0, 0) for v in state): state[0] = 1
        direction = [(rng.randint(-3, 3), rng.randint(-3, 3)) for _ in range(d)]
        add(f'general complex d{d} case{k}', diagonal=diagonal, entries=entries, state=state, direction=direction, denominator=k+1, path=(k-3, k+1))
# Matched invariance cases, without sharing production code.
base = dict(CASES[14][1])
for label, factor in [('raw state scaled', 7), ('negative raw state', -1)]:
    p = dict(base)
    for i in range(p['dimension']):
        for part in ['Real', 'Imag']: p[f'state{part}{i}'] *= factor
    CASES.append((label, p))
p = dict(base)
for i in range(4): p[f'stateReal{i}'], p[f'stateImag{i}'] = -p[f'stateImag{i}'], p[f'stateReal{i}']
CASES.append(('raw state global phase', p))
for label, scale, shift in [('Hamiltonian shift', 1, 13), ('Hamiltonian positive scale', 3, 0), ('Hamiltonian negative scale', -2, 0)]:
    p = dict(base)
    for key in p:
        if key.startswith(('hReal', 'hImag')): p[key] *= scale
    for i in range(p['dimension']): p[f'hReal{i}{i}'] += shift*p['hDenominator']
    CASES.append((label, p))

add('maximum complex coefficients', diagonal=[-1000000,1000000,999999,-999999], entries={(0,1):(1000000,-1000000),(0,2):(-999999,1000000),(0,3):(1000000,999999),(1,2):(-1000000,-1000000),(1,3):(999999,-999999),(2,3):(1000000,-999999)}, denominator=1000000, state=[(1000000,-1000000),(-1000000,999999),(999999,1000000),(1000000,999999)], direction=[(-1000000,999999),(1000000,1000000),(-999999,-1000000),(999999,-999999)], path=(-1000000,1000000))
add('near degenerate separated blocks', diagonal=[1000000,0,1000000,0], entries={(0,1):1,(2,3):2}, denominator=1000000, state=[1,(0,1),2,(1,-1)], direction=[1,2,-1,(0,1)])
add('pure phase direction', diagonal=[-1,3], state=[(2,1),(-1,2)], direction=[(-1,2),(-2,-1)])
add('pure scale direction', diagonal=[-1,3], state=[(2,1),(-1,2)], direction=[(2,1),(-1,2)])
add('zero mean finite variance', diagonal=[-1,1], state=[1,1], direction=[1,-1])
add('extreme negative path', diagonal=[-1000000,1000000], state=[(1000000,1000000),(-1000000,999999)], direction=[(-999999,1000000),(1000000,-1000000)], path=(-1000000,1))
add('complex two degenerate subspaces', diagonal=[1,1,1,1], entries={(0,1):(0,1),(2,3):(0,-1)}, state=[1,(1,-1),2,(0,-1)], direction=[(1,1),-1,(0,1),2])

def exact(x): return str(s.cancel(s.re(x)))
def mpvalue(x): return mp.mpc(str(s.re(x)), str(s.im(x)))
def decimal(x): return 'decimal:' + mp.nstr(mp.re(x), 215)
def matrix_reference(a): return [[{'real':decimal(a[i,j].real), 'imaginary':decimal(a[i,j].imag)} for j in range(a.cols)] for i in range(a.rows)]

def reference(label, p):
    d = p['dimension']
    H = s.zeros(d)
    for i in range(d):
        for j in range(i, d):
            x = (s.Integer(p[f'hReal{i}{j}']) + (s.I*p[f'hImag{i}{j}'] if i != j else 0))/p['hDenominator']
            H[i,j], H[j,i] = x, s.conjugate(x)
    z = s.Matrix([p[f'stateReal{i}']+s.I*p[f'stateImag{i}'] for i in range(d)])
    eta = s.Matrix([p[f'directionReal{i}']+s.I*p[f'directionImag{i}'] for i in range(d)])
    def moments(v):
        N = s.expand((v.H*v)[0])
        if N == 0: return None
        E = s.cancel((v.H*H*v)[0]/N)
        r = (H-E*s.eye(d))*v
        return N,E,s.cancel((v.H*H*H*v)[0]/N),s.cancel((r.H*r)[0]/N),r
    N,E,E2,V,r = moments(z)
    tau = s.Symbol('tau', real=True)
    curve = z+tau*eta
    quotient = s.cancel((curve.H*H*curve)[0]/(curve.H*curve)[0])
    first = s.cancel(s.diff(quotient,tau).subs(tau,0))
    norm_first = s.expand(s.diff((curve.H*curve)[0],tau).subs(tau,0))
    second = s.cancel(s.diff(quotient,tau,2).subs(tau,0))
    t = s.Rational(p['pathNumerator'], p['pathDenominator'])
    path = moments(z+t*eta)
    polynomial = H.charpoly().as_poly()
    intervals = s.polys.polytools.intervals(polynomial, eps=s.Rational(1, 10**230))
    MH = mp.matrix([[mpvalue(H[i,j]) for j in range(d)] for i in range(d)])
    values, vectors = mp.eighe(MH)
    mz = mp.matrix([mpvalue(x) for x in z]); mn = mp.re((mz.H*mz)[0])
    spectra = []; offset=0
    for (a,b), multiplicity in intervals:
        lam = (mp.mpf(str(a.p))/int(a.q)+mp.mpf(str(b.p))/int(b.q))/2
        P = mp.zeros(d)
        for j in range(offset, offset+multiplicity):
            assert abs(values[j]-lam) < mp.mpf('1e-210')*max(1, abs(lam)), label
            v = vectors[:,j]; P += v*v.H
        offset += multiplicity
        prob = mp.re((mz.H*P*mz)[0])/mn
        assert prob >= -mp.mpf('1e-230') and prob <= 1+mp.mpf('1e-230')
        if abs(prob)<mp.mpf('1e-230'): prob=mp.mpf(0)
        if abs(prob-1)<mp.mpf('1e-230'): prob=mp.mpf(1)
        spectra.append({'energy':decimal(lam), 'multiplicity':multiplicity, 'probability':decimal(prob), 'projector':matrix_reference(P)})
    gap = None if len(spectra)==1 else mp.mpf(spectra[1]['energy'][8:])-mp.mpf(spectra[0]['energy'][8:])
    ground_energy = mp.mpf(spectra[0]['energy'][8:]);excess=mp.mpf(str(E.p))/int(E.q)-ground_energy
    overlap = mp.mpf(spectra[0]['probability'][8:])
    return dict(label=label,input=p,expected=dict(dimension=d,rawNormSquared=exact(N),energy=exact(E),secondMoment=exact(E2),variance=exact(V),variation=dict(firstDerivative=exact(first),secondDerivative=exact(second),normFirstDerivative=exact(norm_first)),characteristicPolynomial=[exact(x) for x in polynomial.all_coeffs()],spectrum=spectra,path=dict(defined=path is not None,moments=None if path is None else dict(rawNormSquared=exact(path[0]),energy=exact(path[1]),secondMoment=exact(path[2]),variance=exact(path[3]))),ground=dict(multiplicity=spectra[0]['multiplicity'],energy=decimal(ground_energy),overlap=decimal(overlap),leakage=decimal(1-overlap),excessEnergy=decimal(excess),gap=None if gap is None else decimal(gap),energyOverGap=None if gap is None else decimal(excess/gap))))

result = dict(precisionDecimalDigits=250,reportedDigits=215,method='SymPy exact moments/characteristic/Sturm intervals plus independent mpmath eighe spectral projectors',spectralGrouping='Exact SymPy real-root multiplicities;never floating distance grouping',referenceZeroPolicy='Only fixture probabilities within1e-230 of0/1 are represented at their mathematical endpoint;test tolerance1e-205;does not determine production classification',cases=[reference(label,p) for label,p in CASES])
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(f'{len(CASES)} independent Rayleigh references written to {OUT}')
