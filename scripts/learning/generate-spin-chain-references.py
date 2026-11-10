"""Independent Kronecker/partial-trace references; imports no product modules."""
import json
import random
from pathlib import Path
import sympy as S
import mpmath as mp

mp.mp.dps = 250
axes = 'XYZ'
pauli = {'I': S.eye(2), 'X': S.Matrix([[0, 1], [1, 0]]), 'Y': S.Matrix([[0, -S.I], [S.I, 0]]), 'Z': S.diag(1, -1)}


def graph_edges(n, graph):
    return [(i, j) for i in range(n) for j in range(i + 1, n)
            if graph == 2 or j == i + 1 or graph == 1 and n >= 3 and (i, j) == (0, n - 1)]


def preparation(n, graph=0):
    p = dict(sites=n, graph=graph, hDenominator=1, offset=0)
    p.update({f'field{a}{i}': 0 for i in range(4) for a in axes})
    p.update({f'coupling{a}{b}{i}{j}': 0 for i in range(4) for j in range(i + 1, 4) for a in axes for b in axes})
    p.update({f'state{part}{i}': 0 for i in range(16) for part in ['Real', 'Imag']})
    p['stateReal0'] = 1
    return p


def operator(n, items):
    letters = ['I'] * n
    for i, a in items:
        letters[i] = a
    return S.kronecker_product(*[pauli[a] for a in letters]), ''.join(letters)


def fraction(x):
    value = S.simplify(x)
    assert value.is_Rational, value
    return str(value)


def complex_value(x):
    return {'real': fraction(S.re(x)), 'imaginary': fraction(S.im(x))}


def matrix_value(m):
    return [[complex_value(m[i, j]) for j in range(m.cols)] for i in range(m.rows)]


def divided_root(x, norm):
    def coordinate(v):
        if v == 0:
            return '0'
        exact = S.Rational(v)
        value = mp.mpf(int(exact.p)) / mp.mpf(int(exact.q)) / mp.sqrt(mp.mpf(int(norm.p)) / mp.mpf(int(norm.q)))
        return 'decimal:' + mp.nstr(value, 245)
    return {'real': coordinate(S.re(x)), 'imaginary': coordinate(S.im(x))}


def reference(p):
    n, dimension = p['sites'], 2**p['sites']
    z = S.Matrix([p[f'stateReal{i}'] + S.I * p[f'stateImag{i}'] for i in range(dimension)])
    norm = S.expand((z.H * z)[0])
    H, terms = S.zeros(dimension), []

    def add_term(key, items):
        nonlocal H
        if not p[key]:
            return
        matrix, word = operator(n, items)
        coefficient = S.Rational(p[key], p['hDenominator'])
        mean = S.simplify((z.H * matrix * z)[0] / norm)
        H += coefficient * matrix
        terms.append(dict(sourceKey=key, word=word, support=[i for i, _ in items], coefficient=fraction(coefficient), expectation=fraction(mean), energyContribution=fraction(coefficient * mean)))

    add_term('offset', [])
    for i in range(n):
        for a in axes:
            add_term(f'field{a}{i}', [(i, a)])
    edges = graph_edges(n, p['graph'])
    for i, j in edges:
        for a in axes:
            for b in axes:
                add_term(f'coupling{a}{b}{i}{j}', [(i, a), (j, b)])
    H = H.applyfunc(S.expand)
    action = (H * z).applyfunc(S.expand)
    energy = S.expand((z.H * action)[0] / norm)
    h_squared = (H * H).applyfunc(S.expand)
    second = S.expand((z.H * h_squared * z)[0] / norm)
    variance = S.simplify(second - energy**2)
    residual = action - energy * z
    assert H == H.H and variance >= 0
    rho = (z * z.H / norm).applyfunc(S.expand)
    local_states, local_means = [], []
    for site in range(n):
        # Trace the outer density operator directly; do not infer it from Bloch coordinates.
        reduced = S.zeros(2)
        mask = 2**(n - site - 1)
        for i in range(dimension):
            for j in range(dimension):
                if (i & ~mask) == (j & ~mask):
                    reduced[int(bool(i & mask)), int(bool(j & mask))] += rho[i, j]
        reduced = reduced.applyfunc(S.expand)
        bloch = [S.simplify(S.trace(reduced * pauli[a])) for a in axes]
        purity = S.simplify(S.trace(reduced * reduced))
        assert S.simplify(S.trace(reduced)) == 1 and S.Rational(1, 2) <= purity <= 1
        local_means.append(bloch)
        local_states.append(dict(site=site, bloch=list(map(fraction, bloch)), purity=fraction(purity), pure=purity == 1, density=matrix_value(reduced)))
    correlations = []
    for i in range(n):
        for j in range(i + 1, n):
            values = []
            for ai, a in enumerate(axes):
                row = []
                for bi, b in enumerate(axes):
                    matrix, _ = operator(n, [(i, a), (j, b)])
                    value = S.simplify(S.trace(rho * matrix))
                    row.append(dict(value=fraction(value), connected=fraction(value - local_means[i][ai] * local_means[j][bi])))
                values.append(row)
            correlations.append(dict(sites=[i, j], values=values))

    def symmetry(matrix):
        commutator = (H * matrix - matrix * H).applyfunc(S.expand)
        squared = S.simplify(S.trace(commutator.H * commutator))
        mean = S.simplify(S.trace(rho * matrix))
        spread = S.simplify(S.trace(rho * matrix * matrix) - mean**2)
        return dict(conserved=squared == 0, commutatorFrobeniusSquared=fraction(squared), expectation=fraction(mean), variance=fraction(spread), selectedStateSharp=spread == 0)

    total_z = sum((operator(n, [(i, 'Z')])[0] for i in range(n)), S.zeros(dimension))
    parity = S.kronecker_product(*[pauli['Z']] * n)
    diagonal = H.is_diagonal()
    spectrum = None
    if diagonal:
        energies = list(H.diagonal())
        distinct = sorted(set(energies))
        groups = []
        for e in distinct:
            indices = [i for i, value in enumerate(energies) if value == e]
            probability = S.simplify(sum(rho[i, i] for i in indices))
            groups.append(dict(energy=fraction(e), basisIndices=indices, multiplicity=len(indices), probability=fraction(probability)))
        ground_p = S.Rational(groups[0]['probability'])
        gap = distinct[1] - distinct[0] if len(distinct) > 1 else None
        ratio = S.simplify((energy - distinct[0]) / gap) if gap is not None else None
        spectrum = dict(basisEnergies=list(map(fraction, energies)), levels=groups, groundEnergy=fraction(distinct[0]), groundMultiplicity=groups[0]['multiplicity'], groundProbability=fraction(ground_p), leakage=fraction(1 - ground_p), excitationGap=fraction(gap) if gap is not None else None, excessEnergy=fraction(energy - distinct[0]), excessEnergyToGapRatio=fraction(ratio) if ratio is not None else None, certifiedLeakageUpperBound=fraction(min(1, ratio)) if ratio is not None else None, selectedStateGround=ground_p == 1)
    return dict(sites=n, dimension=dimension, graphEdges=[list(e) for e in edges], activeEdges=[[i, j] for i, j in edges if any(p[f'coupling{a}{b}{i}{j}'] for a in axes for b in axes)], hamiltonian=matrix_value(H), state=[complex_value(x) for x in z], normalizedState=[divided_root(x, norm) for x in z], hamiltonianAction=[complex_value(x) for x in action], rawNormSquared=fraction(norm), energy=fraction(energy), secondMoment=fraction(second), variance=fraction(variance), residual=[complex_value(x) for x in residual], normalizedResidual=[divided_root(x, norm) for x in residual], hamiltonianFrobeniusSquared=fraction(S.trace(H.H * H)), terms=terms, localStates=local_states, correlations=correlations, totalZ=symmetry(total_z), globalZParity=symmetry(parity), spectralMode='basis-enumeration' if diagonal else 'not-computed-for-nondiagonal-hamiltonian', diagonalSpectrum=spectrum, classification=dict(exactEigenstate=variance == 0, selectedStateGround=spectrum['selectedStateGround'] if spectrum is not None else None), cost=dict(complexStateAmplitudes=dimension, denseComplexMatrixEntries=dimension**2, nonzeroPauliTerms=len(terms), nonzeroMatrixEntries=sum(value != 0 for value in H)))


cases = []


def save(label, p):
    snapshot = dict(p)
    cases.append(dict(label=label, input=snapshot, expected=reference(snapshot)))
    print(f'Verified {label}', flush=True)


rng = random.Random(20261109)
for n in range(1, 5):
    for graph in range(3):
        for sample in range(2):
            p = preparation(n, graph)
            p['hDenominator'] = [3, 7][sample]
            p['offset'] = rng.randint(-3, 3)
            for i in range(n):
                for a in axes:
                    p[f'field{a}{i}'] = rng.randint(-3, 3)
            for i, j in graph_edges(n, graph):
                for a in axes:
                    for b in axes:
                        p[f'coupling{a}{b}{i}{j}'] = rng.randint(-3, 3)
            for i in range(2**n):
                p[f'stateReal{i}'], p[f'stateImag{i}'] = rng.randint(-3, 3), rng.randint(-3, 3)
            if not any(p[f'state{part}{i}'] for i in range(2**n) for part in ['Real', 'Imag']):
                p['stateReal0'] = 1
            save(f'general complex {n} sites graph{graph} case{sample}', p)

p = preparation(4); p['hDenominator'] = 2; p['stateReal15'] = 1
for i in range(4):
    p[f'fieldX{i}'] = -1
for i, j in graph_edges(4, 0):
    p[f'couplingZZ{i}{j}'] = -2
save('GHZ4 transverse field', p)
for i in range(4):
    p[f'fieldX{i}'] = 0
save('GHZ4 diagonal ferromagnet', p)
p['stateReal0'] = p['stateReal15'] = 0; p['stateReal5'] = 1
save('zero variance excited ferromagnet', p)
p = preparation(3, 1)
for i, j in graph_edges(3, 1):
    p[f'couplingZZ{i}{j}'] = 1
for i in range(8):
    p[f'stateReal{i}'] = 1
save('frustrated triangle uniform state', p)
p['stateReal0'] = 0
for i in range(1, 8):
    p[f'stateReal{i}'] = 0
p['stateReal7'] = 1
save('frustrated triangle exact excited state', p)
p = preparation(4, 2); p['offset'] = -3; p['stateReal0'] = 0
for i in [1, 2, 4, 8]:
    p[f'stateReal{i}'] = 1
save('W4 constant Hamiltonian', p)
p = preparation(2); p['stateReal0'] = 0; p['stateReal1'] = p['stateReal2'] = 1; p['couplingXX01'] = p['couplingYY01'] = 1
save('XX YY sharp total Z excited eigenstate', p)
p['stateReal2'] = -1
save('XX YY singlet has no supplied ground certificate', p)
p = preparation(2); p['stateReal0'] = 0; p['stateReal1'] = 1; p['stateImag2'] = 1; p['couplingXY01'] = 1; p['couplingYX01'] = -1
save('chiral XY YX complex eigenstate', p)
p = preparation(2); p['stateImag1'] = 1; p['fieldY1'] = 1
save('right local Y phase orientation', p)
p = preparation(1); p['hDenominator'] = 1000000; p['fieldX0'] = 1
save('tiny positive transverse variance', p)
p = preparation(1); p['hDenominator'] = 1000000; p['fieldZ0'] = 1; p['stateReal0'] = 1000000; p['stateReal1'] = 1
save('tiny positive near eigenstate variance', p)
for graph in [0, 1, 2]:
    p = preparation(2, graph); p['couplingZZ01'] = 1
    save(f'two site unique ZZ bond graph{graph}', p)
p = preparation(4, 2); p['couplingXZ03'] = 1; p['stateReal0'] = 1; p['stateImag15'] = -1
save('custom nonneighbor XZ edge', p)
p = preparation(2); p['fieldZ0'] = 1; p['fieldZ1'] = 3
save('gap ratio above probability range', p)
p = preparation(4, 2); p['offset'] = 1000000
for i in range(4):
    for a in axes:
        p[f'field{a}{i}'] = 1000000 if (i + axes.index(a)) % 2 else -1000000
for i, j in graph_edges(4, 2):
    for a in axes:
        for b in axes:
            p[f'coupling{a}{b}{i}{j}'] = 1000000 if (i + j + axes.index(a) + axes.index(b)) % 2 else -1000000
for i in range(16):
    p[f'stateReal{i}'], p[f'stateImag{i}'] = 1000000, -1000000
save('maximum complex integer preparation', p)

target = Path('scripts/fixtures/spin-chain-reference.json')
target.write_text(json.dumps(dict(method='Independent SymPy Kronecker matrices, outer-density partial traces and matrix commutators; mpmath250dps normalized coordinates', seed=20261109, cases=cases), ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(cases=len(cases), dimensions=sorted(set(c['input']['sites'] for c in cases)), graphs=sorted(set(c['input']['graph'] for c in cases)), output=str(target))))
