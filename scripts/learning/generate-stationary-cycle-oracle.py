"""Independent Fraction matrices, path sums, and graph checks for the cycle model."""
from decimal import Decimal, localcontext
from fractions import Fraction as F
from itertools import product
from math import gcd
from pathlib import Path
import json

CASES = [
    ('reset', (1, 1, 1), (1, 0, 0), 8),
    ('biased', (2, 1, 1), (1, 0, 0), 12),
    ('reversed', (1, 2, 1), (1, 0, 0), 12),
    ('symmetric-no-hold', (1, 1, 0), (1, 0, 0), 12),
    ('directed-clockwise', (1, 0, 0), (1, 0, 0), 12),
    ('directed-counterclockwise', (0, 1, 0), (1, 0, 0), 12),
    ('identity-skew', (0, 0, 1), (1, 2, 7), 12),
    ('identity-uniform-zero-step', (0, 0, 1000000), (1, 1, 1), 0),
    ('biased-uniform', (2, 1, 1), (1, 1, 1), 64),
    ('rare-forward', (1, 0, 1000000), (1, 0, 0), 512),
    ('rare-backward', (0, 1, 1000000), (1, 0, 0), 512),
    ('near-reset-underflow', (1000000, 999999, 1000000), (1, 0, 0), 512),
    ('near-reset-reversed', (999999, 1000000, 1000000), (1, 0, 0), 64),
    ('maximum-reset', (1000000, 1000000, 1000000), (1, 0, 0), 512),
    ('zero-step-skew', (2, 1, 1), (1000000, 0, 1), 0),
    ('initial-state-one', (3, 0, 2), (0, 1000000, 0), 32),
    ('weighted-initial', (7, 2, 5), (1, 2, 4), 64),
    ('scaled-transition', (6, 3, 3), (1, 0, 0), 12),
    ('scaled-initial', (2, 1, 1), (1000000, 0, 0), 12),
    ('maximum-directed', (1000000, 0, 0), (1, 0, 0), 511),
    ('tiny-holding', (1000000, 999999, 1), (1, 0, 0), 64),
    ('backward-and-hold', (0, 2, 3), (4, 1, 0), 32),
    ('symmetric-tiny-hold', (1000000, 1000000, 1), (1, 0, 0), 64),
    ('nearly-symmetric-no-hold', (1000000, 999999, 0), (1, 0, 0), 64),
]


def advance(mu, p):
    return [sum(mu[i] * p[i][j] for i in range(3)) for j in range(3)]


def multiply(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def rank(matrix):
    a = [list(row) for row in matrix]
    pivot_row = 0
    for col in range(3):
        found = next((i for i in range(pivot_row, 3) if a[i][col]), None)
        if found is None:
            continue
        a[pivot_row], a[found] = a[found], a[pivot_row]
        scale = a[pivot_row][col]
        a[pivot_row] = [v / scale for v in a[pivot_row]]
        for i in range(3):
            if i != pivot_row:
                factor = a[i][col]
                a[i] = [v - factor * w for v, w in zip(a[i], a[pivot_row])]
        pivot_row += 1
    return pivot_row


def observation(q):
    value = float(q)
    with localcontext() as ctx:
        ctx.prec = 100
        logarithm = str((Decimal(abs(q.numerator)) / Decimal(q.denominator)).ln()) if q else None
    return dict(numerator=str(q.numerator), denominator=str(q.denominator),
                value=None if q and value == 0 else value,
                sign=1 if q > 0 else -1 if q < 0 else 0, logMagnitude=logarithm)


def observations(value):
    if isinstance(value, F):
        return observation(value)
    if isinstance(value, list):
        return [observations(v) for v in value]
    if isinstance(value, dict):
        return {k: observations(v) for k, v in value.items()}
    return value


output = []
path_comparisons = 0
all_step_checks = 0
for name, (forward, backward, stay), initial_weights, steps in CASES:
    total = forward + backward + stay
    # Assemble by moving probability to named destinations, independently of runtime row construction.
    p = [[F(0) for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for offset, weight in [(0, stay), (1, forward), (-1, backward)]:
            p[i][(i + offset) % 3] += F(weight, total)
    pi = [F(1, 3)] * 3
    initial = [F(w, sum(initial_weights)) for w in initial_weights]
    delta = max(sum(abs(p[i][k] - p[j][k]) for k in range(3)) / 2 for i in range(3) for j in range(3))
    flux = [[pi[i] * p[i][j] for j in range(3)] for i in range(3)]
    current = [[flux[i][j] - flux[j][i] for j in range(3)] for i in range(3)]
    assert advance(pi, p) == pi
    stationary_rank = rank([[p[j][i] - F(i == j) for j in range(3)] for i in range(3)])
    reachable = [[bool(p[i][j]) or i == j for j in range(3)] for i in range(3)]
    for k in range(3):
        for i in range(3):
            for j in range(3):
                reachable[i][j] |= reachable[i][k] and reachable[k][j]
    groups = sorted({tuple(j for j in range(3) if reachable[i][j] and reachable[j][i]) for i in range(3)})
    periods = [0, 0, 0]
    power = [[F(i == j) for j in range(3)] for i in range(3)]
    for time in range(1, 10):
        power = multiply(power, p)
        for i in range(3):
            if power[i][i] > 0:
                periods[i] = gcd(periods[i], time)
    irreducible = len(groups) == 1
    assert (stationary_rank == 2) == irreducible
    history = []
    mu = initial
    initial_tv = sum(abs(v - F(1, 3)) for v in initial) / 2
    selected = set(range(5)) | {7, 8, 11, 12, 15, 31, 63, 127, 255, 511, steps}
    for time in range(steps + 1):
        tv = sum(abs(v - F(1, 3)) for v in mu) / 2
        bound = initial_tv * delta ** time
        assert sum(mu) == 1 and min(mu) >= 0 and tv <= bound
        if time <= 5 and total <= 20:
            path_sum = [F(0)] * 3
            for path in product(range(3), repeat=time + 1):
                probability = initial[path[0]]
                for x, y in zip(path, path[1:]):
                    probability *= p[x][y]
                path_sum[path[-1]] += probability
            assert path_sum == mu
            path_comparisons += 3
        if time in selected:
            history.append(dict(step=time, probabilities=mu, deviationFromUniform=[v - F(1, 3) for v in mu],
                                totalVariationFromUniform=tv, contractionUpperBound=bound))
        all_step_checks += 1
        if time != steps:
            mu = advance(mu, p)
    next_mu = advance(mu, p)
    classification = dict(irreducible=irreducible, stationaryReferenceIsUnique=stationary_rank == 2,
                          stationaryFamily='unique-uniform' if stationary_rank == 2 else 'all-distributions',
                          detailedBalanceAtUniform=all(current[i][j] == 0 for i in range(3) for j in range(3)),
                          returnPeriods=periods, uniqueClassPeriod=periods[0] if irreducible else None,
                          communicatingClasses=[list(g) for g in groups],
                          convergesToUniformFromEveryInitial=irreducible and periods[0] == 1,
                          convergesToUniformForThisInitial=initial == pi or irreducible and periods[0] == 1,
                          currentDistributionIsStationary=mu == next_mu)
    reference = dict(transition=p, timeReversedTransition=[list(r) for r in zip(*p)], stationaryReference=pi,
                     stationaryResidual=[v - q for v, q in zip(advance(pi, p), pi)], stationaryFlux=flux, stationaryCurrent=current,
                     stationaryBalance=[dict(state=i, incoming=sum(flux[j][i] for j in range(3) if j != i),
                                             outgoing=sum(flux[i][j] for j in range(3) if j != i), holding=flux[i][i], netOutflow=F(0)) for i in range(3)],
                     clockwiseCurrent=current[0][1], conditionalForwardCycleProbability=p[0][1] * p[1][2] * p[2][0],
                     conditionalBackwardCycleProbability=p[0][2] * p[2][1] * p[1][0],
                     cycleProductDifference=p[0][1] * p[1][2] * p[2][0] - p[0][2] * p[2][1] * p[1][0],
                     contractionCoefficient=delta, history=history, nextDistribution=next_mu,
                     currentFlux=[[mu[i] * p[i][j] for j in range(3)] for i in range(3)],
                     currentNetOutflow=[v - w for v, w in zip(mu, next_mu)])
    output.append(dict(id=name, parameters=dict(forwardWeight=forward, backwardWeight=backward, stayWeight=stay,
                                               initialWeights=initial_weights, stepCount=steps),
                       reference=observations(reference), classification=classification, stationaryRank=stationary_rank))

result = dict(scope='independent exact Fraction reference for actual integer-weight runtime inputs',
              cases=output, caseCount=len(output), allFiniteTimeChecks=all_step_checks,
              enumeratedPathEndpointComparisons=path_comparisons,
              selectedHistoryRows=sum(len(c['reference']['history']) for c in output))
fixture = Path('scripts/fixtures/stationary-cycle-oracle.json')
fixture.write_text(json.dumps(result, indent=2) + '\n')
folder = Path('.runtime/learning/stationary-cycle')
folder.mkdir(parents=True, exist_ok=True)
(folder / 'oracle-verification.json').write_text(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'cases'}))
