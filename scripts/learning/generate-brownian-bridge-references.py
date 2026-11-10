"""Independent finite bridge references: temporal incidence, exact elimination, increment observables."""
from fractions import Fraction as F
from pathlib import Path
import json

DEFAULT = dict(intervals=4, time='1', diffusion='1', start='0', end='1', probe=2, source='0')
CASES = [
 ('four-interval-hand-check', {}), ('one-interior-coordinate', dict(intervals=2, probe=1)),
 ('refined-eight', dict(intervals=8, probe=4)), ('maximum-thirty-two', dict(intervals=32, probe=16)),
 ('double-diffusion', dict(diffusion='2')), ('double-time', dict(time='2')),
 ('translated-endpoints', dict(start='3', end='4')), ('changed-displacement', dict(end='2')),
 ('middle-source', dict(source='2')), ('negative-source', dict(source='-2')),
 ('fixed-start-source', dict(probe=0, start='-2', source='3')), ('fixed-end-source', dict(probe=4, source='16')),
 ('fractional-grid', dict(intervals=7, time='3/2', diffusion='2/3', start='-1/2', end='5/4', probe=3, source='1/3')),
 ('equivalent-spelling', dict(time='2/2', diffusion='3/3', start='0/5', end='4/4', source='0/7')),
 ('minimum-positive-factors', dict(intervals=32, time='1/999999999989', diffusion='1/999999999983', start='-16', end='16', probe=1, source='16')),
 ('maximum-coprime', dict(intervals=32, time='999999999983/62499999999', diffusion='999999999979/62499999999', start='-999999999961/62499999999', end='999999999959/62499999999', probe=17, source='999999999953/62499999999')),
]
CASES += [(f'sweep-{n}', dict(intervals=n, probe=n//2, time='7/3', diffusion='3/5', start='-2', end='4/3', source='-5/7')) for n in range(2, 33)]


def inverse(matrix):
    size = len(matrix)
    rows = [row[:] + [F(i == j) for j in range(size)] for i, row in enumerate(matrix)]
    pivots = []
    for k in range(size):
        pivot = rows[k][k]
        assert pivot > 0
        pivots.append(pivot)
        rows[k] = [value / pivot for value in rows[k]]
        for i in range(size):
            if i != k:
                coefficient = rows[i][k]
                rows[i] = [value - coefficient * other for value, other in zip(rows[i], rows[k])]
    return [row[size:] for row in rows], pivots


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def vector(matrix, values):
    return [dot(row, values) for row in matrix]


def wire(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, list):
        return [wire(x) for x in value]
    if isinstance(value, dict):
        return {k: wire(v) for k, v in value.items()}
    return value


def reference(raw):
    n = raw['intervals']; size = n - 1
    time, diffusion, start, end, source = [F(raw[k]) for k in ['time', 'diffusion', 'start', 'end', 'source']]
    probe = raw['probe']; step = time / n
    incidence = [[F((edge + 1 == i) - (edge == i)) for i in range(1, n)] for edge in range(n)]
    precision = [[sum(row[i]*row[j] for row in incidence) / (2*diffusion*step) for j in range(size)] for i in range(size)]
    inner_cov, pivots = inverse(precision)
    covariance = [[inner_cov[i-1][j-1] if 0 < i < n and 0 < j < n else F(0) for j in range(n+1)] for i in range(n+1)]
    forcing = [F(0) for _ in range(size)]
    forcing[0] += start / (2*diffusion*step)
    forcing[-1] += end / (2*diffusion*step)
    tilted_forcing = forcing[:]
    if 0 < probe < n:
        tilted_forcing[probe - 1] += source
    base_inner = vector(inner_cov, forcing); tilted_inner = vector(inner_cov, tilted_forcing)
    base_mean = [start, *base_inner, end]; mean = [start, *tilted_inner, end]
    log_generating = (dot(tilted_forcing, tilted_inner) - dot(forcing, base_inner))/2
    if probe in [0, n]:
        log_generating += source * base_mean[probe]
    base_action = sum((base_mean[i+1]-base_mean[i])**2 for i in range(n))/(4*diffusion*step)
    determinant = F(1)
    for p in pivots:
        determinant *= p
    prefactor_square_times_pi = (4*diffusion*step)**(-n) * 2**size / determinant
    assert prefactor_square_times_pi == 1/(4*diffusion*time)
    increment_mean = [mean[i+1]-mean[i] for i in range(n)]
    increment_cov = [[covariance[i+1][j+1]-covariance[i+1][j]-covariance[i][j+1]+covariance[i][j] for j in range(n)] for i in range(n)]
    area_weights = [step/2, *[step]*(n-1), step/2]
    area_mean = dot(area_weights, mean)
    area_variance = dot(area_weights, vector(covariance, area_weights))
    quadratic_variation = sum(increment_mean[i]**2 + increment_cov[i][i] for i in range(n))
    limiting_area = diffusion*time**3/6
    rows = []
    for factor in [1, 2, 4, 8]:
        count = n*factor; delta = time/count; q = probe*factor
        # Conditional independent increments: Sigma=2D delta(I - 11^T/count).
        beta = [delta*(count-r+F(1, 2)) for r in range(1, count+1)]
        source_increment = [2*diffusion*delta*(F(r <= q)-F(q, count)) for r in range(1, count+1)]
        mean_increment = [(end-start)/count+source*c for c in source_increment]
        inc_variance = 2*diffusion*delta*(1-F(1, count))
        variation = sum(x*x+inc_variance for x in mean_increment)
        area_var = 2*diffusion*delta*(dot(beta, beta)-sum(beta)**2/count)
        rows.append(dict(intervals=count, integrationDimension=count-1, matchedProbeIndex=q, timeStep=delta,
                         probeTime=q*delta, meanAtProbe=start+sum(mean_increment[:q]),
                         varianceAtProbe=2*diffusion*delta*(q-F(q*q, count)),
                         areaMean=time*start+dot(beta, mean_increment), areaVariance=area_var,
                         areaVarianceGap=limiting_area-area_var, expectedQuadraticVariation=variation,
                         expectedAction=variation/(4*diffusion*delta), fullMatrixCalculated=factor==1))
    result = dict(
        input=raw, integrationDimension=size, timeStep=step, times=[i*step for i in range(n+1)],
        baselineMean=base_mean, mean=mean, precision=precision, covariance=covariance,
        factorization=dict(diagonal=pivots, subdiagonal=[precision[i+1][i]/pivots[i] for i in range(size-1)], determinant=determinant),
        kernel=dict(prefactorSquaredTimesPi=prefactor_square_times_pi, baselineNegativeLogExponential=base_action,
                    tiltedLogExponential=log_generating-base_action, formula='K_tau(b,a)=exp[-(b-a)^2/(4 D tau)]/sqrt(4 pi D tau)'),
        source=dict(convention='dimensionless-exponent-point-source', probeTime=probe*step,
                    baselineMeanAtProbe=base_mean[probe], varianceAtProbe=covariance[probe][probe],
                    logGenerating=log_generating, meanResponse=[row[probe] for row in covariance]),
        area=dict(baselineMean=dot(area_weights, base_mean), mean=area_mean, variance=area_variance,
                  analyticLimitingVariance=limiting_area, varianceGap=limiting_area-area_variance),
        increments=dict(mean=increment_mean, covariance=increment_cov,
                        deterministicQuadraticVariation=dot(increment_mean, increment_mean), expectedQuadraticVariation=quadratic_variation,
                        expectedAction=quadratic_variation/(4*diffusion*step),
                        baselineExpectedAction=sum((base_mean[i+1]-base_mean[i])**2+increment_cov[i][i] for i in range(n))/(4*diffusion*step),
                        fluctuationAction=sum(increment_cov[i][i] for i in range(n))/(4*diffusion*step)),
        refinements=rows,
        certificates=dict(meanResidual=[dot(precision[i], tilted_inner)-tilted_forcing[i] for i in range(size)],
                          inverseResidual=[[sum(precision[i][k]*inner_cov[k][j] for k in range(size))-F(i==j) for j in range(size)] for i in range(size)]),
        scope=dict(fullMatrixIntervals=n, largestAnalyticRefinementIntervals=n*8, continuousMeasureExistenceProved=False,
                   randomPathsGenerated=False, transcendentalDecimalEnclosuresComputed=False),
    )
    assert all(x == 0 for x in result['certificates']['meanResidual'])
    assert all(x == 0 for row in result['certificates']['inverseResidual'] for x in row)
    return wire(result)


fixture = dict(method='Python Fraction temporal incidence / Gauss-Jordan / increment linear functionals; no TypeScript producer imports', cases=[])
for label, patch in CASES:
    raw = dict(DEFAULT, **patch)
    fixture['cases'].append(dict(id=label, input=raw, expected=reference(raw)))
out = Path('scripts/fixtures/brownian-bridge-references.json')
out.write_text(json.dumps(fixture, indent=2)+'\n')
print(json.dumps(dict(cases=len(CASES), bytes=out.stat().st_size, maximumIntervals=32, standardLibraryOnly=True)))
