"""Standard-library reference: incidence/Gauss-Jordan and Gaussian composition.

No TypeScript numerical module is imported or executed. Inputs are reviewed data.
"""
from fractions import Fraction as F
from pathlib import Path
import json
import sys

# Trusted finite reference inputs can produce more than4300 digits at256 slices.
# Keep an explicit ceiling rather than disabling Python's conversion limit.
if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(65536)


def text(value):
    return str(value)


def inverse(matrix):
    n = len(matrix)
    rows = [row[:] + [F(i == j) for j in range(n)] for i, row in enumerate(matrix)]
    pivots = []
    for k in range(n):
        pivot = rows[k][k]
        assert pivot > 0
        pivots.append(pivot)
        rows[k] = [x / pivot for x in rows[k]]
        for i in range(n):
            if i != k and rows[i][k]:
                factor = rows[i][k]
                rows[i] = [x - factor * y for x, y in zip(rows[i], rows[k])]
    return [row[n:] for row in rows], pivots


def determinant(matrix):
    rows = [row[:] for row in matrix]
    value = F(1)
    for k in range(len(rows)):
        pivot = rows[k][k]
        assert pivot > 0
        value *= pivot
        for i in range(k + 1, len(rows)):
            if rows[i][k]:
                factor = rows[i][k] / pivot
                rows[i] = [x - factor * y for x, y in zip(rows[i], rows[k])]
    return value


def convolve(after, before):
    pa, la, ra, ha = after
    pb, lb, rb, hb = before
    total = ra + lb
    return pa * pb / total, la - ha * ha / (4 * total), rb - hb * hb / (4 * total), ha * hb / (2 * total)


def reference(source):
    n, probe = source['intervals'], source['probe']
    tau, mass, hbar, omega, a, b = [F(source[k]) for k in ['time', 'mass', 'hbar', 'omegaSquared', 'start', 'end']]
    size, step = n - 1, tau / n
    scale, potential = mass / (hbar * step), mass * omega * step / hbar
    incidence = [[F((edge + 1 == i) - (edge == i)) for i in range(1, n)] for edge in range(n)]
    fixed = [F(0)] * n
    fixed[0] -= a
    fixed[-1] += b
    precision = [[scale * sum(row[i] * row[j] for row in incidence) + potential * (i == j) for j in range(size)] for i in range(size)]
    forcing = [-scale * sum(row[i] * x for row, x in zip(incidence, fixed)) for i in range(size)]
    interior, pivots = inverse(precision)
    mean = [a] + [sum(x * y for x, y in zip(row, forcing)) for row in interior] + [b]
    cov = [[interior[i - 1][j - 1] if 0 < i < n and 0 < j < n else F(0) for j in range(n + 1)] for i in range(n + 1)]
    det = F(1)
    ds = [F(1)]
    for pivot in pivots:
        det *= pivot
        ds.append(ds[-1] * pivot / scale)
    p = (scale / 2) ** n * 2 ** size / det
    g = scale / 2 + potential / 4 - scale * scale * interior[0][0] / 2
    h = scale * scale * interior[0][-1]
    constant = scale * sum(x * x for x in fixed) / 2 + potential * (a * a + b * b) / 4
    minimum = constant - sum(x * y for x, y in zip(forcing, mean[1:-1])) / 2
    right_shift = potential * (a * a - b * b) / 4
    kinetic = scale * sum((mean[i + 1] - mean[i]) ** 2 + cov[i + 1][i + 1] + cov[i][i] - 2 * cov[i][i + 1] for i in range(n)) / 2
    potential_action = potential * ((a * a + b * b) / 2 + sum(mean[i] ** 2 + cov[i][i] for i in range(1, n))) / 2
    assert kinetic + potential_action == minimum + F(size, 2)
    assert minimum == g * (a * a + b * b) - h * a * b
    if omega:
        periodic = [[F(0) for _ in range(n)] for _ in range(n)]
        for edge in range(n):
            row = [F(0)] * n
            row[edge] -= 1
            row[(edge + 1) % n] += 1
            for i in range(n):
                for j in range(n):
                    periodic[i][j] += scale * row[i] * row[j]
        for i in range(n):
            periodic[i][i] += potential
        assert scale ** n / determinant(periodic) == p / (2 * g - h)
    else:
        assert 2 * g - h == 0
    values, ids = [], {}
    indices = []
    for row in cov:
        current = []
        for value in row:
            key = text(value)
            if key not in ids:
                ids[key] = len(values)
                values.append(key)
            current.append(ids[key])
        indices.append(current)
    refinements = []
    for multiplier in [1, 2, 4, 8]:
        count, q, delta = n * multiplier, probe * multiplier, tau / (n * multiplier)
        s = mass / (hbar * delta)
        diagonal = s / 2 + mass * omega * delta / (4 * hbar)
        cache = {1: (s / 2, diagonal, diagonal, s)}

        def kernel(k):
            if k not in cache:
                half = kernel(k // 2)
                combined = convolve(half, half)
                cache[k] = convolve(cache[1], combined) if k % 2 else combined
            return cache[k]

        pp, gl, gr, hh = kernel(count)
        assert gl == gr
        if 0 < q < count:
            _, left_l, _, left_h = kernel(q)
            _, _, right_r, right_h = kernel(count - q)
            denominator = left_l + right_r
            mu, variance = (left_h * a + right_h * b) / (2 * denominator), 1 / (2 * denominator)
        else:
            mu, variance = (a if q == 0 else b), F(0)
        refinements.append(dict(intervals=count, integrationDimension=count-1, matchedProbeIndex=q, timeStep=text(delta), probeTime=text(q * delta), meanAtProbe=text(mu), varianceAtProbe=text(variance), prefactorSquaredTimesPi=text(pp), diagonalExponentCoefficient=text(gl), crossExponentCoefficient=text(hh), minimumAction=text(gl * b * b + gr * a * a - hh * a * b), rightLogCorrection=text(right_shift / multiplier), traceSquared=text(pp / (gl + gr - hh)) if omega else None, fullMatrixCalculated=multiplier == 1))
    mean_residual = [sum(precision[i][j] * mean[j + 1] for j in range(size)) - forcing[i] for i in range(size)]
    inverse_residual = [[sum(precision[i][k] * interior[k][j] for k in range(size)) - (i == j) for j in range(size)] for i in range(size)]
    assert all(x == 0 for x in mean_residual) and all(x == 0 for row in inverse_residual for x in row)
    return dict(
        input=source, integrationDimension=size, timeStep=text(step), inverseTemperature=text(tau / hbar), times=[text(i * step) for i in range(n + 1)], mean=list(map(text, mean)), precision=[list(map(text, row)) for row in precision], covariance=dict(values=[values[i:i+128] for i in range(0, len(values), 128)], indices=indices), endpointForcing=list(map(text, forcing)),
        factorization=dict(diagonal=list(map(text, pivots)), subdiagonal=[text(precision[i + 1][i] / pivots[i]) for i in range(size - 1)], determinant=text(det), recurrence=list(map(text, ds))),
        kernel=dict(prefactorSquaredTimesPi=text(p), diagonalExponentCoefficient=text(g), crossExponentCoefficient=text(h), actionConstant=text(constant), minimumAction=text(minimum), rightLogCorrection=text(right_shift), rightLogExponential=text(right_shift - minimum), symmetricUnderEndpointReversal=True),
        trace=dict(status='finite' if omega else 'divergent-unbounded-line', diagonalCurvature=text(2 * g - h), squared=text(p / (2 * g - h)) if omega else None),
        endpointIntegral=dict(prefactorSquared=text(p / g), negativeLogQuadraticCoefficient=text(g - h * h / (4 * g)), conservesMass=not omega),
        expectedAction=dict(kinetic=text(kinetic), potential=text(potential_action), total=text(kinetic + potential_action), minimum=text(minimum), fluctuation=text(F(size, 2))),
        refinements=refinements, certificates=dict(meanResidual=list(map(text, mean_residual)), inverseResidual=[list(map(text, row)) for row in inverse_residual]),
        scope=dict(fullMatrixIntervals=n, largestAnalyticRefinementIntervals=n*8, covarianceRepresentation='indexed-exact-fractions', exactFiniteRule='trapezoidal-potential', continuousKernelEvaluated=False, transcendentalDecimalEnclosuresComputed=False, generalOperatorConvergenceProved=False, randomPathsGenerated=False),
    )


def main():
    path = Path('scripts/fixtures/euclidean-kernel-references.json')
    inputs = json.loads(Path('scripts/fixtures/euclidean-kernel-reference-inputs.json').read_text())
    cases = []
    for source in inputs:
        cases.append(dict(id=source['id'], input=source['input'], expected=reference(source['input'])))
        print(source['id'], flush=True)
    path.write_text(json.dumps(dict(method='Python Fraction:incidence/Gauss-Jordan/periodic determinant/Gaussian convolution;no TS numerical imports', cases=cases), indent=2) + '\n')
    print(json.dumps(dict(cases=len(cases), fixtureBytes=path.stat().st_size)))


if __name__ == '__main__':
    main()
