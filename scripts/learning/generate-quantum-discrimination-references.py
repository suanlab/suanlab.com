"""Independent Hermitian-matrix reference, not the product Bloch formula.
Scalar certificates are a+b*sqrt(c), obtained from minimal polynomials.
"""
import json
from functools import lru_cache
from pathlib import Path
import sympy as s

z = s.Symbol('z')
I = s.eye(2)


def density(r):
    x, y, z = map(s.Rational, r)
    return s.Matrix([[1+z, x-s.I*y], [x+s.I*y, 1-z]])/2


@lru_cache(None)
def scalar(expression):
    expression = s.simplify(expression)
    if expression.is_Rational:
        return [str(expression), '0', '0']
    polynomial = s.Poly(s.minpoly(expression, z), z)
    if polynomial.degree() == 1:
        A, B = polynomial.all_coeffs()
        return [str(-B/A), '0', '0']
    assert polynomial.degree() == 2, expression
    A, B, C = polynomial.all_coeffs()
    radicand = B*B-4*A*C
    for sign in (-1, 1):
        a, b = -B/(2*A), s.Rational(sign)/(2*A)
        if s.simplify(expression-a-b*s.sqrt(radicand)) == 0:
            return [str(a), str(b), str(radicand)]
    raise ArithmeticError(f'Could not identify quadratic root: {expression}')


def projector(vector):
    return (vector*vector.conjugate().T/(vector.conjugate().T*vector)[0]).applyfunc(s.simplify)


def root_matrix(rho):
    result = s.zeros(2)
    for value, _, basis in rho.eigenvects():
        for vector in basis:
            result += s.sqrt(value)*projector(vector)
    return result.applyfunc(s.simplify)


@lru_cache(None)
def pair_properties(left, right):
    rho, sigma = density(left), density(right)
    distance = s.simplify(sum(abs(value)*multiplicity for value, multiplicity in (rho-sigma).eigenvals().items())/2)
    root = root_matrix(rho)
    middle = (root*sigma*root).applyfunc(s.simplify)
    fidelity = s.simplify(sum(s.sqrt(value)*multiplicity for value, multiplicity in middle.eigenvals().items())**2)
    return scalar(distance), scalar(fidelity)


def reference(name, left, right, prior):
    left, right, prior = tuple(map(s.Rational, left)), tuple(map(s.Rational, right)), s.Rational(prior)
    rho, sigma = density(left), density(right)
    delta = prior*rho-(1-prior)*sigma
    eigen = delta.eigenvals()
    trace_norm = s.simplify(sum(abs(value)*multiplicity for value, multiplicity in eigen.items()))
    signs = [s.sign(value) for value in eigen]
    indefinite = -1 in signs and 1 in signs
    if delta == s.zeros(2):
        strategy, effect = 'uninformative', I
    elif indefinite:
        strategy, effect = 'spectral-projector', s.zeros(2)
        for value, _, basis in delta.eigenvects():
            if value > 0:
                for vector in basis:
                    effect += projector(vector)
    elif s.trace(delta) >= 0:
        strategy, effect = 'always-zero', I
    else:
        strategy, effect = 'always-one', s.zeros(2)
    probabilities = []
    for state in (rho, sigma):
        p = s.simplify(s.trace(effect*state))
        probabilities.append([scalar(p), scalar(1-p)])
    success = s.simplify((1+trace_norm)/2)
    assert s.simplify(prior*s.trace(effect*rho)+(1-prior)*s.trace((I-effect)*sigma)-success) == 0
    distance, fidelity = pair_properties(left, right)
    return dict(name=name, bloch0=list(map(str, left)), bloch1=list(map(str, right)), prior=str(prior),
                strategy=strategy, uniqueOptimalEffect=delta.det() != 0,
                strictlyBeatsPriorOnly=indefinite,
                scalarCertificates=dict(optimalSuccess=scalar(success), priorOnlySuccess=scalar(max(prior, 1-prior)),
                                        traceNorm=scalar(trace_norm), gainAbovePriorOnly=scalar(success-max(prior, 1-prior)),
                                        traceDistance=distance, squaredUhlmannFidelity=fidelity),
                effect=[[[*scalar(s.re(effect[i, j]))], [*scalar(s.im(effect[i, j]))]] for i in range(2) for j in range(2)],
                conditional=probabilities)


states = {
    'zero': (0, 0, 1), 'one': (0, 0, -1), 'plus': (1, 0, 0), 'plus-i': (0, 1, 0),
    'mixed': (0, 0, 0), 'three-four-five': (s.Rational(24, 25), 0, s.Rational(-7, 25)),
    'inside-a': (s.Rational(1, 5), s.Rational(2, 5), s.Rational(1, 5)),
    'inside-b': (s.Rational(1, 3), s.Rational(-1, 6), s.Rational(1, 4)),
}
cases = []
for name0, left in states.items():
    for name1, right in states.items():
        for prior in [0, s.Rational(1, 5), s.Rational(1, 2), s.Rational(4, 5), 1]:
            cases.append(reference(f'{name0}/{name1}/{prior}', left, right, prior))
D = 10**30
left, right = (0, 0, s.Rational(1, 5)), (0, 0, s.Rational(-1, 5))
for name, prior in [('threshold', s.Rational(3, 5)), ('below-threshold', s.Rational(6*D-1, 10*D)), ('above-threshold', s.Rational(6*D+1, 10*D))]:
    cases.append(reference(name, left, right, prior))
cases.append(reference('tiny-distinct', (0, 0, 1), (0, 0, 1-s.Rational(1, D)), s.Rational(1, 2)))
cases.append(reference('tiny-prior-bias', (0, 0, 0), (0, 0, 0), s.Rational(D+1, 2*D)))
destination = Path('scripts/fixtures/quantum-discrimination-independent.json')
destination.write_text(json.dumps(dict(schemaVersion=1, method='SymPy Hermitian eigenspaces and minimal-polynomial scalar certificates', cases=cases), indent=2)+'\n')
print(json.dumps(dict(cases=len(cases), scalarCache=scalar.cache_info()._asdict(), fixture=str(destination))))
