"""Independent JW Kronecker matrices; no product-module imports."""
import json
import random
from pathlib import Path
import sympy as S
import mpmath as mp

mp.mp.dps = 250
I, Z = S.eye(2), S.diag(1, -1)
a = S.Matrix([[0, 1], [0, 0]])


def preparation(modes, sector=1):
    p = dict(modes=modes, sector=sector, hDenominator=1, offset=0)
    p.update({f'onsite{i}': 0 for i in range(4)})
    p.update({f'{prefix}{part}{i}{j}': 0 for prefix in ['hopping', 'pairing'] for i in range(4) for j in range(i + 1, 4) for part in ['Real', 'Imag']})
    p.update({f'density{i}{j}': 0 for i in range(4) for j in range(i + 1, 4)})
    p.update(leftMode=0, rightMode=min(1, modes - 1), leftCreation=1, rightCreation=1)
    p.update({f'state{part}{i}': 0 for i in range(16) for part in ['Real', 'Imag']})
    p[f'stateReal{1 if sector == 2 else 0}'] = 1
    return p


def fraction(x):
    x = S.simplify(x)
    assert x.is_Rational, x
    return str(x)


def complex_value(x):
    return dict(real=fraction(S.re(x)), imaginary=fraction(S.im(x)))


def matrix_value(m):
    return [[complex_value(m[i, j]) for j in range(m.cols)] for i in range(m.rows)]


def normalized_value(x, norm):
    def coordinate(v):
        if v == 0:
            return '0'
        r = S.Rational(v)
        value = mp.mpf(int(r.p)) / mp.mpf(int(r.q)) / mp.sqrt(mp.mpf(int(norm.p)) / mp.mpf(int(norm.q)))
        return 'decimal:' + mp.nstr(value, 245)
    return dict(real=coordinate(S.re(x)), imaginary=coordinate(S.im(x)))


def squared(m):
    return S.expand(S.trace(m.H * m))


def reference(p):
    modes, d = p['modes'], 2**p['modes']
    annihilators = [S.kronecker_product(*[Z if j < i else a if j == i else I for j in range(modes)]) for i in range(modes)]
    naive = [S.kronecker_product(*[a if j == i else I for j in range(modes)]) for i in range(modes)]
    creators = [c.H for c in annihilators]
    numbers = [creators[i] * annihilators[i] for i in range(modes)]
    z = S.Matrix([p[f'stateReal{i}'] + S.I * p[f'stateImag{i}'] for i in range(d)])
    norm = S.expand((z.H * z)[0])
    rho = (z * z.H / norm).applyfunc(S.expand)
    expect = lambda operator: S.expand(S.trace(rho * operator))
    H, naive_h, terms = S.zeros(d), S.zeros(d), []

    def add_term(kind, indices, coefficient, operator, naive_operator, paired=False):
        nonlocal H, naive_h
        if coefficient == 0:
            return
        H += coefficient * operator
        naive_h += coefficient * naive_operator
        if paired:
            H += S.conjugate(coefficient) * operator.H
            naive_h += S.conjugate(coefficient) * naive_operator.H
        mean = expect(operator)
        contribution = 2 * S.re(coefficient * mean) if paired else coefficient * mean
        terms.append(dict(type=kind, modes=indices, coefficient=complex_value(coefficient), expectation=complex_value(mean), includesAdjoint=paired, energyContribution=fraction(contribution)))

    denominator = p['hDenominator']
    add_term('offset', [], S.Rational(p['offset'], denominator), S.eye(d), S.eye(d))
    for i in range(modes):
        add_term('onsite', [i], S.Rational(p[f'onsite{i}'], denominator), numbers[i], numbers[i])
    for i in range(modes):
        for j in range(i + 1, modes):
            hopping = (p[f'hoppingReal{i}{j}'] + S.I * p[f'hoppingImag{i}{j}']) / denominator
            pairing = (p[f'pairingReal{i}{j}'] + S.I * p[f'pairingImag{i}{j}']) / denominator
            add_term('hopping', [i, j], hopping, creators[i] * annihilators[j], naive[i].H * naive[j], True)
            add_term('pairing', [i, j], pairing, creators[i] * creators[j], naive[i].H * naive[j].H, True)
            add_term('density', [i, j], S.Rational(p[f'density{i}{j}'], denominator), numbers[i] * numbers[j], numbers[i] * numbers[j])
    H, naive_h = H.applyfunc(S.expand), naive_h.applyfunc(S.expand)
    assert H == H.H and naive_h == naive_h.H
    action = (H * z).applyfunc(S.expand)
    energy = expect(H)
    second = expect((H * H).applyfunc(S.expand))
    variance = S.simplify(second - energy**2)
    residual = (action - energy * z).applyfunc(S.expand)
    assert variance >= 0
    occupations = [i.bit_count() for i in range(d)]
    parity_values = [(-1)**n for n in occupations]
    number_operator = sum(numbers, S.zeros(d))
    parity_operator = S.kronecker_product(*[Z] * modes)
    twirl = ((rho + parity_operator * rho * parity_operator) / 2).applyfunc(S.expand)
    coherence = squared(rho - twirl)
    gamma = S.Matrix(modes, modes, lambda i, j: expect(creators[i] * annihilators[j]))
    anomalous = S.Matrix(modes, modes, lambda i, j: expect(annihilators[i] * annihilators[j]))
    assert gamma == gamma.H and anomalous == -anomalous.T
    for mask in range(1, 2**modes):
        indices = [i for i in range(modes) if mask & (1 << i)]
        assert S.simplify(gamma.extract(indices, indices).det()) >= 0
        assert S.simplify((S.eye(modes) - gamma).extract(indices, indices).det()) >= 0
    car = []
    for i in range(modes):
        for j in range(modes):
            mixed = annihilators[i] * creators[j] + creators[j] * annihilators[i] - (S.eye(d) if i == j else S.zeros(d))
            aa = annihilators[i] * annihilators[j] + annihilators[j] * annihilators[i]
            dd = creators[i] * creators[j] + creators[j] * creators[i]
            assert mixed == aa == dd == S.zeros(d)
            car.append(dict(leftMode=i, rightMode=j, mixedResidualFrobeniusSquared=fraction(squared(mixed)), annihilationResidualFrobeniusSquared=fraction(squared(aa)), creationResidualFrobeniusSquared=fraction(squared(dd))))

    def symmetry(operator):
        commutator = (H * operator - operator * H).applyfunc(S.expand)
        mean = expect(operator)
        spread = S.simplify(expect(operator * operator) - mean**2)
        return dict(conserved=commutator == S.zeros(d), commutatorFrobeniusSquared=fraction(squared(commutator)), expectation=fraction(mean), variance=fraction(spread), selectedStateSharp=spread == 0)

    left = creators[p['leftMode']] if p['leftCreation'] else annihilators[p['leftMode']]
    right = creators[p['rightMode']] if p['rightCreation'] else annihilators[p['rightMode']]
    nl = naive[p['leftMode']].H if p['leftCreation'] else naive[p['leftMode']]
    nr = naive[p['rightMode']].H if p['rightCreation'] else naive[p['rightMode']]
    forward, reverse = (left * right * z).applyfunc(S.expand), (right * left * z).applyfunc(S.expand)
    scalar = int(p['leftMode'] == p['rightMode'] and p['leftCreation'] != p['rightCreation'])
    spectrum = None
    if H.is_diagonal():
        energies = list(H.diagonal())
        distinct = sorted(set(energies))
        groups = []
        for energy_value in distinct:
            indices = [i for i, value in enumerate(energies) if value == energy_value]
            groups.append(dict(energy=fraction(energy_value), basisIndices=indices, multiplicity=len(indices), probability=fraction(sum(rho[i, i] for i in indices))))
        ground = distinct[0]
        ground_probability = S.Rational(groups[0]['probability'])
        gap = distinct[1] - ground if len(distinct) > 1 else None
        ratio = S.simplify((energy - ground) / gap) if gap is not None else None

        def minimum(indices):
            values = sorted(set(energies[i] for i in indices))
            first = values[0]
            basis = [i for i in indices if energies[i] == first]
            probability = sum(rho[i, i] for i in basis)
            return dict(energy=fraction(first), multiplicity=len(basis), basisIndices=basis, probability=fraction(probability), excitationGap=fraction(values[1] - first) if len(values) > 1 else None, selectedStateGround=probability == 1)

        parity_minima = [dict(parity='even' if k == 0 else 'odd', **minimum([i for i, number in enumerate(occupations) if number % 2 == k])) for k in range(2)]
        number_minima = [dict(number=k, **minimum([i for i, number in enumerate(occupations) if number == k])) for k in range(modes + 1)]
        spectrum = dict(basisEnergies=list(map(fraction, energies)), levels=groups, groundEnergy=fraction(ground), groundMultiplicity=groups[0]['multiplicity'], groundProbability=fraction(ground_probability), leakage=fraction(1-ground_probability), excitationGap=fraction(gap) if gap is not None else None, excessEnergyToGapRatio=fraction(ratio) if ratio is not None else None, certifiedLeakageUpperBound=fraction(min(1, ratio)) if ratio is not None else None, selectedStateGround=ground_probability == 1, scope='full-Fock-space-not-a-fixed-number-or-parity-sector', parityMinima=parity_minima, numberMinima=number_minima, selectedParityGround=None if p['sector'] == 0 else parity_minima[p['sector']-1]['selectedStateGround'])
    even_probability = S.simplify(sum(rho[i, i] for i in range(d) if occupations[i] % 2 == 0))
    correlations = []
    for i in range(modes):
        for j in range(i + 1, modes):
            joint = expect(numbers[i] * numbers[j])
            correlations.append(dict(modes=[i, j], joint=fraction(joint), connected=fraction(joint - expect(numbers[i]) * expect(numbers[j]))))
    columns = []
    for mode, matrix in enumerate(annihilators):
        entries = []
        for j in range(d):
            rows = [i for i in range(d) if matrix[i, j] != 0]
            entries.append(dict(row=rows[0] if rows else None, sign=int(matrix[rows[0], j]) if rows else 0))
        columns.append(dict(mode=mode, annihilationColumns=entries))
    naive_energy, naive_second = expect(naive_h), expect((naive_h * naive_h).applyfunc(S.expand))
    return dict(modes=modes, dimension=d, occupationCounts=occupations, hamiltonian=matrix_value(H), state=[complex_value(x) for x in z], normalizedState=[normalized_value(x, norm) for x in z], rawNormSquared=fraction(norm), hamiltonianAction=[complex_value(x) for x in action], energy=fraction(energy), secondMoment=fraction(second), variance=fraction(variance), residual=[complex_value(x) for x in residual], normalizedResidual=[normalized_value(x, norm) for x in residual], terms=terms, density=matrix_value(rho), gamma=matrix_value(gamma), anomalous=matrix_value(anomalous), oddMeans=[complex_value(expect(c)) for c in annihilators], modeOccupations=[fraction(expect(n)) for n in numbers], numberDistribution=[fraction(sum(rho[i, i] for i in range(d) if occupations[i] == k)) for k in range(modes + 1)], numberCorrelations=correlations, totalNumber=symmetry(number_operator), fermionParity=symmetry(parity_operator), parityAssumption=dict(declaredInput='formal-either-parity-allowed' if p['sector'] == 0 else 'even-pure-sector' if p['sector'] == 1 else 'odd-pure-sector', evenProbability=fraction(even_probability), oddProbability=fraction(1-even_probability), coherenceFrobeniusSquared=fraction(coherence), pureStateCompatibleWithParitySuperselection=coherence == 0), parityDephased=dict(density=matrix_value(twirl), purity=fraction(S.expand(S.trace(twirl * twirl))), energy=fraction(S.expand(S.trace(twirl * H))), oddMeans=[complex_value(S.expand(S.trace(twirl * c))) for c in annihilators]), operators=columns, car=car, probe=dict(expectedAnticommutatorScalar=scalar, forward=[complex_value(x) for x in forward], reverse=[complex_value(x) for x in reverse], anticommutatorAction=[complex_value(x) for x in forward + reverse], forwardOnNormalizedInput=[normalized_value(x, norm) for x in forward], reverseOnNormalizedInput=[normalized_value(x, norm) for x in reverse], forwardSquaredNormOnNormalizedInput=fraction(S.expand((forward.H * forward)[0]/norm)), reverseSquaredNormOnNormalizedInput=fraction(S.expand((reverse.H * reverse)[0]/norm)), carResidualFrobeniusSquared=fraction(squared(left*right+right*left-scalar*S.eye(d))), naiveCarResidualFrobeniusSquared=fraction(squared(nl*nr+nr*nl-scalar*S.eye(d)))), naiveComparison=dict(energy=fraction(naive_energy), variance=fraction(naive_second-naive_energy**2), hamiltonianDifferenceFrobeniusSquared=fraction(squared(H-naive_h))), spectralMode='occupation-basis-enumeration' if H.is_diagonal() else 'not-computed-for-nondiagonal-hamiltonian', diagonalSpectrum=spectrum, classification=dict(exactEigenstate=variance == 0, selectedStateGround=spectrum['selectedStateGround'] if spectrum is not None else None, selectedStateParitySectorGround=spectrum['selectedParityGround'] if spectrum is not None else None))


cases = []


def save(label, p):
    snapshot = dict(p)
    assert len(snapshot) == 74
    cases.append(dict(label=label, input=snapshot, expected=reference(snapshot)))
    print('Verified ' + label, flush=True)


rng = random.Random(20261109)
for modes in range(1, 5):
    for sector in range(3):
        for sample in range(2):
            p = preparation(modes, sector)
            p['hDenominator'], p['offset'] = [3, 7][sample], rng.randint(-3, 3)
            p['leftMode'], p['rightMode'] = rng.randrange(modes), rng.randrange(modes)
            p['leftCreation'], p['rightCreation'] = rng.randrange(2), rng.randrange(2)
            for i in range(modes):
                p[f'onsite{i}'] = rng.randint(-3, 3)
                for j in range(i + 1, modes):
                    for prefix in ['hopping', 'pairing']:
                        for part in ['Real', 'Imag']:
                            p[f'{prefix}{part}{i}{j}'] = rng.randint(-3, 3)
                    p[f'density{i}{j}'] = rng.randint(-3, 3)
            for i in range(2**modes):
                allowed = sector == 0 or i.bit_count() % 2 == sector - 1
                p[f'stateReal{i}'], p[f'stateImag{i}'] = (rng.randint(-3, 3), rng.randint(-3, 3)) if allowed else (0, 0)
            if not any(p[f'state{part}{i}'] for i in range(2**modes) for part in ['Real', 'Imag']):
                p[f'stateReal{1 if sector == 2 else 0}'] = 1
            save(f'general complex {modes} modes sector{sector} case{sample}', p)

p = preparation(2); save('creation order on vacuum', p)
p['leftMode'] = p['rightMode'] = 0; p['leftCreation'] = 0; p['rightCreation'] = 1
save('same mode mixed CAR on vacuum', p)
p['leftCreation'] = 1; save('same mode exclusion on vacuum', p)
p = preparation(2, 2); p['stateReal2'] = 1; p['hoppingReal01'] = -1
save('one particle symmetric hopping eigenstate', p)
p['hoppingReal01'] = 0; p['hoppingImag01'] = 1; p['stateReal2'] = 0; p['stateImag2'] = 1
save('imaginary hopping phase', p)
p = preparation(2); p['stateReal3'] = 1; p['pairingReal01'] = 1
save('even pair plus eigenstate', p)
p['stateReal3'] = -1; save('even pair minus eigenstate', p)
p['stateReal3'] = 0; p['stateImag3'] = 1; p['pairingReal01'] = 0; p['pairingImag01'] = 1
save('complex pairing adjoint order', p)
p = preparation(3); p['stateReal0'] = 0; p['stateReal3'] = p['stateReal6'] = 1; p['hoppingReal02'] = 1; p['rightMode'] = 2; p['rightCreation'] = 0
save('long range hopping through occupied middle mode', p)
p = preparation(1, 0); p['stateReal1'] = 1; p['onsite0'] = 1
save('formal even odd coherent one mode', p)
p = preparation(3, 0); p['stateReal7'] = 1
for i in range(3):
    p[f'onsite{i}'] = i + 1
save('formal cat has zero first odd moments', p)
p = preparation(1, 1); p['onsite0'] = -1
save('even sector minimum differs from full Fock minimum', p)
p = preparation(1, 2); p['onsite0'] = -1
save('odd sector full ground state', p)
p = preparation(2); p['pairingReal01'] = 1
save('sharp number is not number conservation', p)
p = preparation(2); p['stateReal3'] = 1; p['onsite0'] = 1; p['onsite1'] = 2
save('number conservation does not imply sharp number', p)
p = preparation(4, 0); p['offset'] = -3
for i in range(16):
    p[f'stateReal{i}'] = 1
save('constant Hamiltonian all parity sectors', p)
p = preparation(2, 2); p['hoppingReal01'] = 1; p['hDenominator'] = 1000000
save('tiny positive hopping variance', p)
p = preparation(2, 2); p['onsite0'] = 1; p['onsite1'] = -1; p['hDenominator'] = 1000000; p['stateReal1'] = 1000000; p['stateReal2'] = 1
save('tiny positive near occupation eigenstate variance', p)
p = preparation(4); p['stateReal0'] = 0; p['stateReal3'] = p['stateReal12'] = 1
save('fixed two particles fractional gamma occupations', p)
p['stateReal3'] = p['stateReal12'] = 0; p['stateReal0'] = p['stateReal15'] = 1
save('even cat same gamma different number distribution', p)
p = preparation(4, 0); p['offset'] = 1000000
for i in range(4):
    p[f'onsite{i}'] = 1000000 if i % 2 else -1000000
    for j in range(i + 1, 4):
        p[f'density{i}{j}'] = -1000000
        for prefix in ['hopping', 'pairing']:
            p[f'{prefix}Real{i}{j}'], p[f'{prefix}Imag{i}{j}'] = 1000000, -1000000
for i in range(16):
    p[f'stateReal{i}'], p[f'stateImag{i}'] = 1000000, -1000000
save('maximum complex integer Fock preparation', p)

target = Path('scripts/fixtures/fermion-fock-reference.json')
target.write_text(json.dumps(dict(method='Independent SymPy Jordan-Wigner Kronecker operators, H^2 moments, density parity twirl and matrix commutators; mpmath250dps normalized coordinates', seed=20261109, cases=cases), ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(cases=len(cases), output=str(target))))
