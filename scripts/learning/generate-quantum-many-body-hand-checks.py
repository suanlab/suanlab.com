"""Independent exact hand examples for the next topic; not a production model."""
import json
from pathlib import Path
import sympy as S

I = S.eye(2)
X = S.Matrix([[0, 1], [1, 0]])
Y = S.Matrix([[0, -S.I], [S.I, 0]])
Z = S.diag(1, -1)


def local(operator, site, count):
    return S.kronecker_product(*[operator if j == site else I for j in range(count)])


def expect(state, operator):
    return S.simplify((state.conjugate().T * operator * state)[0] / state.conjugate().dot(state))


def encoded(value):
    return str(S.simplify(value))


count = 4
state = S.zeros(2**count, 1)
state[0] = state[-1] = 1
hamiltonian = -sum((local(Z, j, count) * local(Z, j + 1, count) for j in range(count - 1)), S.zeros(16))
hamiltonian -= S.Rational(1, 2) * sum((local(X, j, count) for j in range(count)), S.zeros(16))
energy = expect(state, hamiltonian)
variance = expect(state, hamiltonian**2) - energy**2
assert energy == -3 and variance == 1
correlation = expect(state, local(Z, 0, count) * local(Z, 1, count))
assert correlation == 1 and expect(state, local(Z, 0, count)) == 0
ring = sum((local(Z, i, 3) * local(Z, j, 3) for i, j in [(0, 1), (1, 2), (0, 2)]), S.zeros(8))
assert sorted(ring.diagonal()) == [-1] * 6 + [3] * 2

a = S.Matrix([[0, 1], [0, 0]])
fermions = [S.kronecker_product(a, I), S.kronecker_product(Z, a)]
for i in range(2):
    for j in range(2):
        assert fermions[i] * fermions[j].H + fermions[j].H * fermions[i] == (S.eye(4) if i == j else S.zeros(4))
        assert fermions[i] * fermions[j] + fermions[j] * fermions[i] == S.zeros(4)
vacuum = S.Matrix([1, 0, 0, 0])
forward = fermions[0].H * fermions[1].H * vacuum
reverse = fermions[1].H * fermions[0].H * vacuum
assert forward == -reverse == S.Matrix([0, 0, 0, 1])
naive = S.kronecker_product(I, a)
bad = fermions[0] * naive.H + naive.H * fermions[0]
assert S.trace(bad.H * bad) == 4
hopping = -(fermions[0].H * fermions[1] + fermions[1].H * fermions[0])
number = sum((c.H * c for c in fermions), S.zeros(4))
assert hopping * number == number * hopping
assert hopping.eigenvals() == {-1: 1, 0: 2, 1: 1}

cutoff = 3
boson = S.zeros(cutoff + 1)
for n in range(1, cutoff + 1):
    boson[n - 1, n] = S.sqrt(n)
commutator = boson * boson.H - boson.H * boson
assert commutator == S.diag(1, 1, 1, -3)
assert S.trace(commutator) == 0
assert expect(S.Matrix([1, 0, 0, 1]), commutator) == -1

tensors = [
    [S.Matrix([[1, 0]]), S.Matrix([[0, 1]])],
    [S.diag(1, 0), S.diag(0, 1)],
    [S.Matrix([1, 0]), S.Matrix([0, 1])],
]


def contract(network):
    values = []
    for index in range(8):
        bits = [(index >> shift) & 1 for shift in [2, 1, 0]]
        values.append(S.simplify((network[0][bits[0]] * network[1][bits[1]] * network[2][bits[2]])[0]))
    return S.Matrix(values)


gauge = S.Matrix([[2, 1], [0, 1]])
changed = [[v * gauge for v in tensors[0]], [gauge.inv() * v for v in tensors[1]], tensors[2]]
assert contract(tensors) == contract(changed) == S.Matrix([1, 0, 0, 0, 0, 0, 0, 1])
parameter_norm = lambda network: S.simplify(sum(S.trace(v.H * v) for site in network for v in site))
assert parameter_norm(tensors) == 6 and parameter_norm(changed) == S.Rational(19, 2)

coefficients = S.diag(3, 2, 1, 0)
rho = coefficients * coefficients.H / 14
weights = [S.Rational(9, 14), S.Rational(4, 14), S.Rational(1, 14), S.Integer(0)]
assert sorted(rho.eigenvals()) == sorted(weights)
discarded = sum(weights[2:])
assert discarded == S.Rational(1, 14)
normalized_distance_squared = 2 * (1 - S.sqrt(1 - discarded))
assert S.simplify(normalized_distance_squared - (2 - S.sqrt(182) / 7)) == 0
complex_coefficients = S.Matrix([[1, S.I], [0, 0]])
right_density = complex_coefficients.T * complex_coefficients.conjugate() / 2
coefficient_gram = complex_coefficients.H * complex_coefficients / 2
assert S.trace(right_density * Y) == 1 and S.trace(coefficient_gram * Y) == -1

report = {
    "stage": "independent-authoring-hand-checks-only",
    "productionModelsPrepared": False,
    "readerRegistered": False,
    "basisConvention": "n0 is the most significant occupation bit; rightmost operator acts first; spin operators are Pauli matrices",
    "spin": {"sites": 4, "rawNormSquared": 2, "energy": encoded(energy), "variance": encoded(variance), "connectedZZ": encoded(correlation), "triangleEnergies": list(map(int, sorted(ring.diagonal())))},
    "fermion": {"allTwoModeCARVerified": True, "naiveAnticommutatorFrobeniusSquared": 4, "creationOrderSigns": [1, -1], "hoppingSpectrum": [-1, 0, 0, 1], "numberConserved": True},
    "boson": {"cutoff": cutoff, "commutatorDiagonal": [1, 1, 1, -3], "commutatorTrace": 0, "boundarySuperpositionExpectation": -1},
    "mps": {"denseState": list(map(int, contract(tensors))), "gaugeDeterminant": 2, "parameterNormBefore": "6", "parameterNormAfter": "19/2", "physicalRawNormSquared": "2", "stateUnchanged": True},
    "schmidt": {"weights": list(map(encoded, weights)), "middleCutRank": 3, "rankTwoDiscardedWeight": encoded(discarded), "normalizedFidelitySquared": "13/14", "normalizedDistanceSquared": encoded(normalized_distance_squared), "degenerateRankOneErrorSquared": "1/2", "degenerateBestProjectorUnique": False},
    "complexPartialTrace": {"rightDensityYExpectation": 1, "coefficientGramYExpectation": -1, "sameSpectraDoNotEstablishSameDensityOperator": True},
}
target = Path('.runtime/learning/quantum-variational-integration/next-topic-hand-check.json')
target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False))
