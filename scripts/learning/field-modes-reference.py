"""Independent incidence-matrix spectra and 450-digit projection references.
No production calculation or trigonometric implementation is imported.
SymPy checks the variational Hessian; mpmath diagonalizes the finite matrix.
Numerical comparison allowances are not a formal interval proof.
"""
import json
from pathlib import Path
import sympy as sp
import mpmath as mp
mp.mp.dps = 450
R = sp.Rational
ROOT = Path(__file__).resolve().parents[2]

def wire(x):
    return str(x)

def num(x):
    return mp.mpf(str(sp.numer(x))) / mp.mpf(str(sp.denom(x)))

def decimal(x):
    return mp.nstr(x, 340)

def grid(boundary, n):
    pairs = [(i, (i+1) % n) for i in range(n)] if boundary == 'periodic' else [(None if i == 0 else i-1, None if i == n else i) for i in range(n+1)] if boundary == 'dirichlet-zero' else [(i, i+1) for i in range(n-1)]
    incidence = sp.zeros(len(pairs), n)
    for row, (l, r) in enumerate(pairs):
        if l is not None:
            incidence[row, l] -= 1
        if r is not None:
            incidence[row, r] += 1
    return incidence

def definitions(boundary, n):
    if boundary == 'periodic':
        out = [(0, 'constant')]
        for m in range(1, (n-1)//2+1):
            out += [(m, 'cosine'), (m, 'sine')]
        return out + ([(n//2, 'nyquist')] if n % 2 == 0 else [])
    return [(m+1, 'sine') if boundary == 'dirichlet-zero' else (m, 'cosine' if m else 'constant') for m in range(n)]

def raw_shape(boundary, n, m, family):
    if family == 'constant':
        return [mp.mpf(1)] * n
    if family == 'nyquist':
        return [mp.mpf((-1)**j) for j in range(n)]
    return [(mp.sin(mp.pi*m*(j+1)/(n+1)) if boundary == 'dirichlet-zero' else mp.cos(mp.pi*m*(j+mp.mpf('.5'))/n) if boundary == 'neumann-zero' else (mp.sin if family == 'sine' else mp.cos)(2*mp.pi*m*j/n)) for j in range(n)]

def rayleigh(boundary, n, m, family, h, c, mu):
    # Construct energy from individual oriented differences, then divide by mass norm.
    u = raw_shape(boundary, n, m, 'cosine' if family == 'nyquist' and m*2 != n else family)
    edges = [u[(i+1) % n]-u[i] for i in range(n)] if boundary == 'periodic' else [u[0]] + [u[i+1]-u[i] for i in range(n-1)] + [-u[-1]] if boundary == 'dirichlet-zero' else [u[i+1]-u[i] for i in range(n-1)]
    return mu + c*c/(h*h)*mp.fsum(e*e for e in edges)/mp.fsum(x*x for x in u)

symbolic = []
for boundary in ['periodic', 'dirichlet-zero', 'neumann-zero']:
    for n in [3, 4, 5]:
        q = sp.Matrix(sp.symbols('q:'+str(n)))
        h, c, a, b, bg = sp.symbols('h c a b bg', nonzero=True)
        E = grid(boundary, n)
        U = c*c/(2*h)*((E*q).dot(E*q)) + h*sum(a*x*x/2+b*x**4/4 for x in q)
        H = sp.hessian(U, list(q)).subs(dict.fromkeys(q, bg))
        expected = c*c/h*(E.T*E) + h*(a+3*b*bg*bg)*sp.eye(n)
        assert (H-expected).applyfunc(sp.simplify) == sp.zeros(n)
        symbolic.append({'boundary': boundary, 'sites': n, 'energyHessianChecked': True})

cases = []
for boundary in ['periodic', 'dirichlet-zero', 'neumann-zero']:
    for n in [3, 4, 5, 8, 16, 32]:
        for tag, a, b, bg, selected in [('free', '0', '0', '0', n-1), ('saddle', '-2', '0', '0', 0), ('well', '-1', '1', '1', n//2), ('nonstationary', '1/3', '2/5', '1/2', 1)]:
            cases.append({'label': f'{boundary}/{n}/{tag}', 'input': {'boundary': boundary, 'length': '3/2', 'speed': '5/3', 'quadratic': a, 'quartic': b, 'background': bg, 'position': [wire(R((i*5)%11-5, i+2)) for i in range(n)], 'velocity': [wire(R((i*3)%7-3, i+3)) for i in range(n)], 'selectedBasis': selected}})
cases += json.loads((ROOT/'scripts/fixtures/field-modes-preparations.json').read_text())
for count, case in enumerate(cases):
    p = case['input']; boundary = p['boundary']; n = len(p['position']); fixed = boundary == 'dirichlet-zero'
    L, c, a, b, bg = [R(p[k]) for k in ['length', 'speed', 'quadratic', 'quartic', 'background']]
    h = L/(n+1 if fixed else n); q = sp.Matrix([R(x) for x in p['position']]); v = sp.Matrix([R(x) for x in p['velocity']]); reference = sp.ones(n, 1)*bg; eta = q-reference
    E = grid(boundary, n); D = E.T*E; mu = a+3*b*bg*bg; A = c*c/(h*h)*D + mu*sp.eye(n)
    potential = lambda z: c*c/(2*h)*(E*z).dot(E*z) + h*sum(a*x*x/2+b*x**4/4 for x in z)
    kinetic = h*v.dot(v)/2; qp = h*(eta.T*A*eta)[0]/2; acc = -c*c/(h*h)*D*reference-sp.ones(n, 1)*(a*bg+b*bg**3); linear = -h*acc.dot(eta); remainder = potential(q)-potential(reference)-linear-qp
    exact = {'perturbationMassNorm': h*eta.dot(eta), 'velocityMassNorm': h*v.dot(v), 'kineticEnergy': kinetic, 'quadraticPotentialEnergy': qp, 'quadraticEnergy': kinetic+qp, 'referencePotential': potential(reference), 'linearTerm': linear, 'nonlinearRemainder': remainder, 'fullEnergy': kinetic+potential(q), 'expansionResidual': R(0)}
    matrix = mp.matrix([[num(A[i,j]) for j in range(n)] for i in range(n)])
    eigen = list(mp.eigsy(matrix, eigvals_only=True))
    definitions_ = definitions(boundary, n); modes = []; bases = []
    for index, (wave, family) in enumerate(definitions_):
        u = raw_shape(boundary, n, wave, family); normalizer = mp.sqrt(num(h)*mp.fsum(x*x for x in u)); u = [x/normalizer for x in u]; bases.append(u)
        Q = num(h)*mp.fsum(x*num(y) for x,y in zip(u,eta)); V = num(h)*mp.fsum(x*num(y) for x,y in zip(u,v))
        lam = eigen[index]
        if abs(lam) < mp.mpf('1e-350'):
            assert A.det() == 0
            state = 'zero-curvature'
        else:
            state = 'oscillatory' if lam > 0 else 'unstable'
        modes.append({'index': index, 'wave': wave, 'family': family, 'classification': state, 'eigenvalue': decimal(lam), 'coordinate': decimal(Q), 'velocity': decimal(V), 'quadraticEnergy': decimal((V*V+lam*Q*Q)/2), 'massNorm': decimal(num(h)*mp.fsum(x*x for x in u))})
    selected = p['selectedBasis']; wave, family = definitions_[selected]; u = bases[selected]
    refinements = []
    for level in range(4):
        sites = (n+1)*2**level-1 if fixed else n*2**level; spacing = L/(sites+1 if fixed else sites); k = (2 if boundary == 'periodic' else 1)*mp.pi*wave/num(L); lam = rayleigh(boundary, sites, wave, family, num(spacing), num(c), num(mu)); cont = num(mu)+num(c*c)*k*k; gap = cont-lam
        refinements.append({'level': level, 'sites': sites, 'spacing': wire(spacing), 'discreteEigenvalue': decimal(lam), 'continuumEigenvalue': decimal(cont), 'waveOperatorGap': decimal(gap), 'relativeWaveOperatorGap': decimal(gap/(num(c*c)*k*k)) if wave else '0', 'secondOrderUpperBound': decimal(num(c*c)*k**4*num(spacing)**2/12), 'masslessPhaseSpeedRatio': decimal(mp.sqrt((lam-num(mu))/(num(c*c)*k*k))) if wave else None})
    case['expected'] = {'exact': {k: wire(x) for k,x in exact.items()}, 'spacing': wire(h), 'effectiveCurvature': wire(mu), 'referenceIsEquilibrium': acc == sp.zeros(n,1), 'referenceAcceleration': [wire(x) for x in acc], 'interpretation': 'frozen-hessian-nonstationary-reference' if acc != sp.zeros(n,1) else 'exact-linear-dynamics' if b == 0 else 'small-perturbations-about-equilibrium', 'modes': modes, 'selectedShape': [decimal(x) for x in u], 'refinements': refinements}
    print(f'{count+1}/{len(cases)} {case["label"]}', flush=True)
result = {'method': 'Independent incidence-energy matrices,SymPy variational Hessians,mpmath symmetric eigensolver and direct mass-inner-product projections; refinement Rayleigh quotients from edge energies', 'decimalPrecision': mp.mp.dps, 'comparisonRelativeAbsoluteAllowance': '1e-320', 'allowanceIsFormalIntervalCertificate': False, 'sympyVersion': sp.__version__, 'symbolicHessianModels': symbolic, 'cases': cases}
(ROOT/'scripts/fixtures/field-modes-independent.json').write_text(json.dumps(result, indent=2)+'\n')
