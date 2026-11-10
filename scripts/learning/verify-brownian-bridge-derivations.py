"""Exact finite time-slicing identities; no claim of a continuum path measure proof."""
from fractions import Fraction as F
from pathlib import Path
import json

checks = []

def equal(label, left, right):
    assert left == right, (label, left, right)
    checks.append(label)


def inverse_and_determinant(matrix):
    n = len(matrix)
    rows = [[F(x) for x in row] + [F(i == j) for j in range(n)] for i, row in enumerate(matrix)]
    determinant = F(1)
    for k in range(n):
        pivot = next(i for i in range(k, n) if rows[i][k])
        if pivot != k:
            rows[pivot], rows[k] = rows[k], rows[pivot]
            determinant = -determinant
        diagonal = rows[k][k]
        determinant *= diagonal
        rows[k] = [v / diagonal for v in rows[k]]
        for i in range(n):
            if i != k:
                factor = rows[i][k]
                rows[i] = [v - factor * w for v, w in zip(rows[i], rows[k])]
    return [row[n:] for row in rows], determinant


def model(n, time, diffusion, start, end):
    delta = time / n
    # Incidence rows describe each temporal edge, with both fixed endpoints omitted.
    incidence = [[F((edge + 1 == site) - (edge == site)) for site in range(1, n)] for edge in range(n)]
    precision = [[sum(row[i] * row[j] for row in incidence) / (2 * diffusion * delta) for j in range(n - 1)] for i in range(n - 1)]
    mean = [start + (end - start) * F(i, n) for i in range(1, n)]
    covariance = [[2 * diffusion * delta * (min(i, j) - F(i * j, n)) for j in range(1, n)] for i in range(1, n)]
    return delta, precision, mean, covariance


for n in range(2, 33):
    for time, diffusion, start, end in [(F(1), F(1), F(0), F(0)), (F(3, 2), F(2, 3), F(-1, 2), F(5, 4))]:
        tag = f'N={n},tau={time},D={diffusion}'
        delta, precision, mean, covariance = model(n, time, diffusion, start, end)
        inverse, determinant = inverse_and_determinant(precision)
        equal(tag + ':incidence-inverse', inverse, covariance)
        equal(tag + ':determinant', determinant, n / (2 * diffusion * delta) ** (n - 1))
        rhs = [F(0) for _ in mean]
        rhs[0] += start / (2 * diffusion * delta)
        rhs[-1] += end / (2 * diffusion * delta)
        equal(tag + ':endpoint-mean', [sum(q * m for q, m in zip(row, mean)) for row in precision], rhs)
        # The integrated prefactor squared, with its single common pi^-1 removed.
        equal(tag + ':kernel-normalization-square', (4 * diffusion * delta) ** (-n) * 2 ** (n - 1) / determinant, 1 / (4 * diffusion * time))
        path_mean = [start, *mean, end]
        mean_action = sum((y - x) ** 2 for x, y in zip(path_mean, path_mean[1:])) / (4 * diffusion * delta)
        equal(tag + ':endpoint-exponent', mean_action, (end - start) ** 2 / (4 * diffusion * time))
        equal(tag + ':area-mean', delta * ((start + end) / 2 + sum(mean)), time * (start + end) / 2)
        equal(tag + ':area-variance', delta ** 2 * sum(map(sum, covariance)), diffusion * time ** 3 * (1 - F(1, n * n)) / 6)
        def cov(i, j):
            return covariance[i - 1][j - 1] if 0 < i < n and 0 < j < n else F(0)
        increment_variances = [cov(i + 1, i + 1) + cov(i, i) - 2 * cov(i, i + 1) for i in range(n)]
        equal(tag + ':every-increment-variance', increment_variances, [2 * diffusion * delta * (1 - F(1, n))] * n)
        quadratic_variation = sum((path_mean[i + 1] - path_mean[i]) ** 2 + increment_variances[i] for i in range(n))
        equal(tag + ':quadratic-variation', quadratic_variation, (end - start) ** 2 / n + 2 * diffusion * time * (1 - F(1, n)))
        equal(tag + ':expected-action', quadratic_variation / (4 * diffusion * delta), mean_action + F(n - 1, 2))

for n in range(2, 17):
    args = (F(7, 3), F(3, 5), F(-2), F(4, 3))
    _, coarse, coarse_mean, coarse_cov = model(n, *args)
    _, fine, fine_mean, fine_cov = model(2 * n, *args)
    retained = list(range(1, 2 * n - 1, 2))
    removed = list(range(0, 2 * n - 1, 2))
    equal(f'N={n}:projective-mean', [fine_mean[i] for i in retained], coarse_mean)
    equal(f'N={n}:projective-covariance', [[fine_cov[i][j] for j in retained] for i in retained], coarse_cov)
    equal(f'N={n}:odd-block-diagonal', [[fine[i][j] for j in removed] for i in removed], [[fine[i][i] if i == j else F(0) for j in removed] for i in removed])
    schur = [[fine[i][j] - sum(fine[i][k] * fine[k][j] / fine[k][k] for k in removed) for j in retained] for i in retained]
    equal(f'N={n}:schur-complement', schur, coarse)
    # A_N and A_2N are different quadrature observables, despite exact marginal consistency.
    coarse_area = args[1] * args[0] ** 3 * (1 - F(1, n * n)) / 6
    fine_area = args[1] * args[0] ** 3 * (1 - F(1, 4 * n * n)) / 6
    equal(f'N={n}:area-refinement-gap', fine_area - coarse_area, args[1] * args[0] ** 3 / (8 * n * n))

out = Path('.runtime/learning/brownian-bridge-foundation')
out.mkdir(parents=True, exist_ok=True)
summary = {'status': 'passed', 'arithmetic': 'Python Fraction, incidence construction and exact Gauss-Jordan elimination', 'checksPassed': len(checks), 'finiteModels': 62, 'coarseningPairs': 15, 'maximumIntervals': 32, 'manuscriptsAuthored': False, 'productionCalculatorImplemented': False, 'continuumMeasureExistenceProvedByThisScript': False, 'checks': checks}
(out / 'derivations.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({key: value for key, value in summary.items() if key != 'checks'}))
