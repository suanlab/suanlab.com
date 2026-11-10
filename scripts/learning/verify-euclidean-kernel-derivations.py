"""Independent finite Gaussian identities; Decimal comparisons are not certified bounds."""
from decimal import Decimal, localcontext
from fractions import Fraction as F
from pathlib import Path
import json


checks = []


def equal(label, left, right):
    assert left == right, (label, left, right)
    checks.append(label)


def inverse_determinant(matrix):
    size = len(matrix)
    rows = [list(row) + [F(i == j) for j in range(size)] for i, row in enumerate(matrix)]
    determinant = F(1)
    for k in range(size):
        pivot = next(i for i in range(k, size) if rows[i][k])
        if pivot != k:
            rows[k], rows[pivot] = rows[pivot], rows[k]
            determinant = -determinant
        value = rows[k][k]
        determinant *= value
        rows[k] = [x / value for x in rows[k]]
        for i in range(size):
            if i != k:
                factor = rows[i][k]
                rows[i] = [x - factor * y for x, y in zip(rows[i], rows[k])]
    return [row[size:] for row in rows], determinant


def recurrence(n, v):
    values = [F(1), 2 + v]
    for _ in range(2, n):
        values.append((2 + v) * values[-1] - values[-2])
    return values


def finite(n, tau, mass, hbar, omega2, a, b):
    delta = tau / n
    scale, v = mass / (hbar * delta), omega2 * delta * delta
    ds = recurrence(n, v)
    determinant = ds[n - 1]
    g = scale / 2 * (1 + v / 2 - ds[n - 2] / determinant)
    h = scale / determinant
    prefactor_square_without_pi = scale / (2 * determinant)
    return g, h, prefactor_square_without_pi, g * (a * a + b * b) - h * a * b


parameters = [
    (F(1), F(1), F(1), F(0), F(-1, 3), F(4, 3)),
    (F(1), F(1), F(1), F(1), F(-1, 3), F(4, 3)),
    (F(3, 2), F(2, 3), F(5, 4), F(7, 5), F(-1, 2), F(5, 4)),
]
models = []
for n in range(2, 33):
    for tau, mass, hbar, omega2, a, b in parameters:
        tag = f'N={n},tau={tau},m={mass},hbar={hbar},omega2={omega2}'
        delta = tau / n
        scale, v, size = mass / (hbar * delta), omega2 * delta * delta, n - 1
        # Construct each kinetic edge independently, retaining the endpoint forcing.
        incidence = [[F((edge + 1 == site) - (edge == site)) for site in range(1, n)] for edge in range(n)]
        fixed = [F(0)] * n
        fixed[0] -= a
        fixed[-1] += b
        q = [[scale * (sum(row[i] * row[j] for row in incidence) + v * (i == j)) for j in range(size)] for i in range(size)]
        force = [-scale * sum(row[i] * value for row, value in zip(incidence, fixed)) for i in range(size)]
        covariance, determinant = inverse_determinant(q)
        mean = [sum(x * y for x, y in zip(row, force)) for row in covariance]
        ds = recurrence(n, v)
        predicted_covariance = [[ds[min(i, j) - 1] * ds[size - max(i, j)] / (scale * ds[size]) for j in range(1, n)] for i in range(1, n)]
        equal(tag + ':inverse-from-incidence', covariance, predicted_covariance)
        equal(tag + ':determinant', determinant, scale ** size * ds[size])
        equal(tag + ':endpoint-mean', mean, [(a * ds[size - i] + b * ds[i - 1]) / ds[size] for i in range(1, n)])
        constant = scale * (a * a + b * b) / 2 + scale * v * (a * a + b * b) / 4
        minimum = constant - sum(x * y for x, y in zip(force, mean)) / 2
        g, h, prefactor, exponent = finite(n, tau, mass, hbar, omega2, a, b)
        equal(tag + ':completed-square', minimum, exponent)
        equal(tag + ':normalization-square', (scale / 2) ** n * 2 ** size / determinant, prefactor)
        path = [a, *mean, b]
        action = scale / 2 * sum((y - x) ** 2 for x, y in zip(path, path[1:])) + scale * v / 2 * ((a * a + b * b) / 2 + sum(x * x for x in mean))
        equal(tag + ':minimum-direct-action', action, minimum)
        # Right endpoint rather than trapezoidal potential differs only at the ends.
        right_action = scale / 2 * sum((y - x) ** 2 for x, y in zip(path, path[1:])) + scale * v / 2 * sum(x * x for x in path[1:])
        equal(tag + ':primitive-endpoint-shift', right_action - action, scale * v * (b * b - a * a) / 4)
        equal(tag + ':reversal-symmetric', finite(n, tau, mass, hbar, omega2, b, a)[3], exponent)
        if not omega2:
            equal(tag + ':free-prefactor', prefactor, mass / (2 * hbar * tau))
            equal(tag + ':free-action', minimum, mass * (b - a) ** 2 / (2 * hbar * tau))
            equal(tag + ':free-endpoint-mass', (prefactor / g, g - h * h / (4 * g)), (F(1), F(0)))
            equal(tag + ':unconfined-trace-diverges', 2 * g - h, F(0))
        else:
            # Independently close the chain: integrate also its common endpoint.
            # N=2 has two parallel temporal edges, so add contributions explicitly.
            periodic = [[F(0) for _ in range(n)] for _ in range(n)]
            for edge in range(n):
                row = [F(0)] * n
                row[edge] -= 1
                row[(edge + 1) % n] += 1
                for i in range(n):
                    for j in range(n):
                        periodic[i][j] += scale * row[i] * row[j]
            for i in range(n):
                periodic[i][i] += scale * v
            _, periodic_determinant = inverse_determinant(periodic)
            trace_square = scale ** n / periodic_determinant
            equal(tag + ':trace-closed-incidence', trace_square, prefactor / (2 * g - h))
            cosine = (1 + v / 2) * ds[n - 1] - ds[n - 2]
            equal(tag + ':trace-chebyshev', trace_square, 1 / (2 * (cosine - 1)))
            assert 0 < prefactor / g < 1 and g - h * h / (4 * g) > 0
            checks.append(tag + ':positive-kernel-is-not-conservative')
        models.append({'intervals': n, 'omega2': str(omega2), 'minimumAction': str(minimum), 'prefactorSquareWithoutPi': str(prefactor)})


def dec(value):
    return Decimal(value.numerator) / Decimal(value.denominator)


def sinh(value):
    return (value.exp() - (-value).exp()) / 2


def cosh(value):
    return (value.exp() + (-value).exp()) / 2


comparisons = []
with localcontext() as context:
    context.prec = 80
    tau = mass = hbar = omega2 = F(1)
    a, b = F(1, 3), F(4, 3)
    z = Decimal(1)
    continuum_prefactor = 1 / (2 * sinh(z))
    continuum_action = ((dec(a * a + b * b) * cosh(z)) - 2 * dec(a * b)) / (2 * sinh(z))
    previous = None
    for n in [4, 8, 16, 32, 64, 128, 256]:
        g, h, p, action = finite(n, tau, mass, hbar, omega2, a, b)
        log_symmetric_ratio = (dec(p) / continuum_prefactor).ln() / 2 - dec(action) + continuum_action
        primitive_shift = -dec((b * b - a * a) / (4 * n))
        log_primitive_ratio = log_symmetric_ratio + primitive_shift
        u = Decimal(1) / n
        theta = 2 * ((1 + u * u / 4).sqrt() + u / 2).ln()
        # Decimal cross-check of exact recurrence against a separate hyperbolic form.
        ds = recurrence(n, F(1, n * n))
        hyperbolic_s = n * sinh(theta) / sinh(n * theta)
        comparisons_to_check = {
            'determinant': (dec(ds[n - 1]), sinh(n * theta) / sinh(theta)),
            'prefactorSquareWithoutPi': (dec(p), hyperbolic_s / 2),
            'diagonalExponentCoefficient': (dec(g), hyperbolic_s * cosh(n * theta) / 2),
            'crossExponentCoefficient': (dec(h), hyperbolic_s),
            'trace': ((dec(p) / dec(2 * g - h)).sqrt(), 1 / (2 * sinh(n * theta / 2))),
        }
        residuals = {key: abs(left / right - 1) for key, (left, right) in comparisons_to_check.items()}
        assert all(value < Decimal('1e-70') for value in residuals.values()), residuals
        row = {'intervals': n, 'logSymmetricKernelRatio': str(log_symmetric_ratio), 'logRightKernelRatio': str(log_primitive_ratio), 'hyperbolicRelativeResiduals': {key: str(value) for key, value in residuals.items()}}
        if previous:
            symmetric_ratio = abs(previous[0] / log_symmetric_ratio)
            right_ratio = abs(previous[1] / log_primitive_ratio)
            assert Decimal('3.8') < symmetric_ratio < Decimal('4.2')
            assert Decimal('1.8') < right_ratio < Decimal('2.2')
            row.update(symmetricErrorHalvingRatio=str(symmetric_ratio), rightErrorHalvingRatio=str(right_ratio))
        previous = log_symmetric_ratio, log_primitive_ratio
        comparisons.append(row)

report = {'unitId': 'euclidean-quantum-kernels-and-time-slicing', 'exactChecksPassed': len(checks), 'finiteModels': len(models), 'exactIntervalRange': [2, 32], 'decimalPrecision': 80, 'decimalIdentityChecks': 5 * len(comparisons), 'decimalComparisonRows': comparisons, 'decimalComparisonsAreCertifiedBounds': False, 'productionCalculatorChecked': False, 'manuscriptsChecked': False, 'readerRegistered': False, 'models': models, 'checks': checks}
output = Path('.runtime/learning/euclidean-kernel-foundation')
output.mkdir(parents=True, exist_ok=True)
(output / 'derivations.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ['exactChecksPassed', 'finiteModels', 'decimalPrecision', 'decimalComparisonsAreCertifiedBounds', 'readerRegistered']}))
