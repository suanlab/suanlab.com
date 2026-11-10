"""Exact rational teaching preparations, independent of future runtime models."""
from fractions import Fraction as F
from pathlib import Path
import json


def advance(distribution, matrix):
    return [sum(distribution[i] * matrix[i][j] for i in range(len(matrix))) for j in range(len(matrix))]


def cycle(forward, backward, stay):
    total = forward + backward + stay
    assert total > 0
    return [[F(stay if i == j else forward if j == (i + 1) % 3 else backward, total)
             for j in range(3)] for i in range(3)]


def residual(pi, matrix):
    return [v - p for v, p in zip(advance(pi, matrix), pi)]


records = []
for weights in [(2, 1, 1), (1, 1, 0), (1, 0, 0), (0, 0, 1), (0, 1, 1), (1, 2, 3), (0, 1, 0), (0, 3, 2)]:
    matrix = cycle(*weights)
    pi = [F(1, 3)] * 3
    flux = [[pi[i] * matrix[i][j] for j in range(3)] for i in range(3)]
    current = F(weights[0] - weights[1], 3 * sum(weights))
    assert residual(pi, matrix) == [0, 0, 0]
    assert all(flux[i][(i + 1) % 3] - flux[(i + 1) % 3][i] == current for i in range(3))
    assert all(sum(flux[i][j] - flux[j][i] for j in range(3)) == 0 for i in range(3))
    assert (current == 0) == (weights[0] == weights[1])
    records.append(dict(family='stationary-cycle', weights=weights, pi=pi, matrix=matrix, current=current))

naive_counterexample = None
for target in [(1, 2, 4), (1, 1, 1), (4, 2, 1), (1, 8, 1), (100, 1, 3)]:
    pi = [F(w, sum(target)) for w in target]
    for weights in [(2, 1, 1), (1, 1, 0), (1, 0, 0), (0, 0, 1), (0, 1, 1)]:
        proposal = cycle(*weights)
        matrix = [[F(0) for _ in range(3)] for _ in range(3)]
        naive = [[F(0) for _ in range(3)] for _ in range(3)]
        for i in range(3):
            for j in range(3):
                if i != j:
                    # Pair flux construction handles one-way proposals without division by zero.
                    matrix[i][j] = min(pi[i] * proposal[i][j], pi[j] * proposal[j][i]) / pi[i]
                    naive[i][j] = proposal[i][j] * min(F(1), pi[j] / pi[i])
            matrix[i][i] = 1 - sum(matrix[i])
            naive[i][i] = 1 - sum(naive[i])
        assert all(sum(row) == 1 and all(p >= 0 for p in row) for row in matrix)
        assert residual(pi, matrix) == [0, 0, 0]
        assert all(pi[i] * matrix[i][j] == pi[j] * matrix[j][i] for i in range(3) for j in range(3))
        if weights in [(1, 0, 0), (0, 0, 1)]:
            assert matrix == [[F(int(i == j)) for j in range(3)] for i in range(3)]
        if target == (1, 2, 4) and weights == (2, 1, 1):
            naive_counterexample = residual(pi, naive)
            assert any(naive_counterexample)
        records.append(dict(family='hastings-correction', target=target, proposalWeights=weights,
                            pi=pi, matrix=matrix, naiveStationarityResidual=residual(pi, naive)))

finite_comparisons = 0
for a, b in [(F(0), F(0)), (F(1), F(1)), (F(1, 4), F(1, 2)), (F(0), F(1, 2)),
             (F(1, 2), F(0)), (F(1, 2), F(1, 2)), (F(1, 100), F(1, 100)), (F(99, 100), F(99, 100))]:
    matrix = [[1 - a, a], [b, 1 - b]]
    for initial in [F(0), F(1, 3), F(1)]:
        distribution = [1 - initial, initial]
        pi_one = a / (a + b) if a + b else None
        if pi_one is not None:
            pi = [1 - pi_one, pi_one]
            assert residual(pi, matrix) == [0, 0]
            assert pi[0] * a == pi[1] * b
        for time in range(33):
            expected = initial if pi_one is None else pi_one + (initial - pi_one) * (1 - a - b) ** time
            assert distribution[1] == expected
            if pi_one is not None:
                tv = sum(abs(p - q) for p, q in zip(distribution, [1 - pi_one, pi_one])) / 2
                assert tv == abs(initial - pi_one) * abs(1 - a - b) ** time
            distribution = advance(distribution, matrix)
            finite_comparisons += 1
        records.append(dict(family='two-state-convergence', a=a, b=b, initialOne=initial,
                            stationaryOne=pi_one, identityFamily=pi_one is None))

report = dict(scope='ideal rational hand preparations only; no runtime model, manuscript or reader completion',
              cycleCases=8, hastingsCases=25, twoStatePreparations=24,
              finiteTimeComparisons=finite_comparisons, naiveCounterexampleResidual=naive_counterexample,
              records=records, registered=False, modelImplemented=False, curriculumComplete=False)
folder = Path('.runtime/learning/detailed-balance-preparation')
folder.mkdir(parents=True, exist_ok=True)
(folder / 'hand-checks.json').write_text(json.dumps(report, default=str, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'records'}, default=str))
