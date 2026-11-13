"""Independent finite-model checks for path information, sampling aliases and regeneration.

Uses only the standard library. The path calculation enumerates potential-event
state sequences and integrates holding exposures through Dirichlet gap moments.
Its omitted-score bound is analytic; floating-point arithmetic is not a directed
rounding certificate. Regenerative comparisons use exact rational arithmetic.
"""
from fractions import Fraction as F
from itertools import product
from math import exp, expm1, cos, pi, sqrt, factorial
import json


def expected_exposures(alpha, beta, time, initial_one=0.0):
    rate = alpha+beta
    zero = beta/rate*time + ((1-initial_one)-beta/rate)*(-expm1(-rate*time))/rate
    return zero, time-zero


def matrix_product(a, b):
    return [[sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def ring_uniformization(a, b, time):
    clock = a+b+0.5
    transition = [[0.0]*3 for _ in range(3)]
    for i in range(3):
        transition[i][i] = 1-(a+b)/clock
        transition[i][(i+1)%3] = a/clock
        transition[i][(i-1)%3] = b/clock
    power = [[float(i == j) for j in range(3)] for i in range(3)]
    weight = exp(-clock*time)
    result = [[weight*x for x in row] for row in power]
    for n in range(1, 121):
        power = matrix_product(power, transition)
        weight *= clock*time/n
        for i in range(3):
            for j in range(3):
                result[i][j] += weight*power[i][j]
    return result


def solve_fraction(a, b):
    augmented = [list(row)+[value] for row, value in zip(a, b)]
    for j in range(len(a)):
        k = next(k for k in range(j, len(a)) if augmented[k][j])
        augmented[j], augmented[k] = augmented[k], augmented[j]
        pivot = augmented[j][j]
        augmented[j] = [x/pivot for x in augmented[j]]
        for k in range(len(a)):
            if k != j:
                factor = augmented[k][j]
                augmented[k] = [x-factor*y for x, y in zip(augmented[k], augmented[j])]
    return [row[-1] for row in augmented]


def verify_path_information():
    cases, max_error, max_bound = 0, 0.0, 0.0
    for a, b, time, initial_one in product([.5, 1., 2.], [.5, 1., 2.], [.025, .05, .1, .2], [0., .5, 1.]):
        clock, cutoff = 4.0, 12
        x = clock*time
        transition = ((1-a/clock, a/clock), (b/clock, 1-b/clock))
        # probability, last state, zero-state segment count, two real-jump counts
        paths = [(1-initial_one, 0, 1, 0, 0), (initial_one, 1, 0, 0, 0)]
        paths = [row for row in paths if row[0]]
        weight, moments = exp(-x), [0.0]*5
        for n in range(cutoff+1):
            segments = n+1
            assert abs(sum(row[0] for row in paths)-1) < 2e-12
            for probability, last, m0, c01, c10 in paths:
                m1 = segments-m0
                t0, t1 = time*m0/segments, time*m1/segments
                t00 = time*time*m0*(m0+1)/(segments*(segments+1))
                t11 = time*time*m1*(m1+1)/(segments*(segments+1))
                t01 = time*time*m0*m1/(segments*(segments+1))
                terms = [c01-a*t0, c10-b*t1,
                         c01*c01-2*a*c01*t0+a*a*t00,
                         c10*c10-2*b*c10*t1+b*b*t11,
                         c01*c10-b*c01*t1-a*c10*t0+a*b*t01]
                moments = [v+weight*probability*t for v, t in zip(moments, terms)]
            if n < cutoff:
                paths = [(p*transition[last][next_state], next_state,
                          m0+(next_state == 0), c01+(last == 0 and next_state == 1),
                          c10+(last == 1 and next_state == 0))
                         for p, last, m0, c01, c10 in paths for next_state in [0, 1]
                         if transition[last][next_state]]
                weight *= x/(n+1)
        # For k+1>x, positive Poisson terms have ratio at most x/(k+1).
        def tail(k):
            return exp(-x)*x**k/factorial(k)/(1-x/(k+1))
        c = max(a, b)*time
        second_bound = x*x*tail(cutoff-1)+(1+2*c)*x*tail(cutoff)+c*c*tail(cutoff+1)
        first_bound = x*tail(cutoff)+c*tail(cutoff+1)
        exposure0, exposure1 = expected_exposures(a, b, time, initial_one)
        targets = [0., 0., a*exposure0, b*exposure1, 0.]
        bounds = [first_bound, first_bound, second_bound, second_bound, second_bound]
        assert all(abs(value-target) <= bound+3e-12
                   for value, target, bound in zip(moments, targets, bounds))
        cases += 1
        max_error = max(max_error, max(abs(v-t) for v, t in zip(moments, targets)))
        max_bound = max(max_bound, second_bound)
    return {'pathFisherPreparations': cases, 'maxPathMomentDifference': max_error,
            'maxAnalyticOmittedScoreBound': max_bound}


def verify_regeneration():
    comparisons, count_comparisons = 0, 0
    rates = [F(1, 4), F(1, 2), F(1), F(2), F(4)]
    for r in product(rates, repeat=3):
        cycle = sum(1/x for x in r)
        stationary = [1/x/cycle for x in r]
        generator = [[F(0)]*3 for _ in range(3)]
        for i in range(3):
            generator[i][i] = -r[i]
            generator[i][(i+1)%3] = r[i]
        for target in range(3):
            centered = [F(i == target)-stationary[target] for i in range(3)]
            h = solve_fraction([[-x for x in generator[0]], [-x for x in generator[1]], stationary],
                               [centered[0], centered[1], F(0)])
            assert all(sum(-generator[i][k]*h[k] for k in range(3)) == centered[i] for i in range(3))
            assert sum(stationary[i]*h[i] for i in range(3)) == 0
            poisson = 2*sum(stationary[i]*centered[i]*h[i] for i in range(3))
            renewal = sum(centered[i]**2/r[i]**2 for i in range(3))/cycle
            assert poisson == renewal
            comparisons += 1
        # Differentiate the tilted generator's characteristic polynomial at zero.
        rate_product = r[0]*r[1]*r[2]
        first = r[0]*r[1]+r[0]*r[2]+r[1]*r[2]
        second = 2*sum(r)
        mean = 3*rate_product/first
        variance = (9*rate_product-second*mean**2)/first
        assert mean == 3/cycle
        assert variance == 9*sum(1/x**2 for x in r)/cycle**3
        assert 1 <= variance/mean < 3
        count_comparisons += 1
    return {'renewalPoissonComparisons': comparisons, 'jumpCountTiltComparisons': count_comparisons}


def verify_aliases():
    preparations = 0
    for rate_sum, interval in product([2., 3., 4.], [2., 3., 4.]):
        difference = 2*pi/(sqrt(3)*interval)
        a, b = (rate_sum+difference)/2, (rate_sum-difference)/2
        assert a > 0 and b > 0
        full = [ring_uniformization(a, b, interval), ring_uniformization(b, a, interval)]
        half = [ring_uniformization(a, b, interval/2), ring_uniformization(b, a, interval/2)]
        assert max(abs(full[0][i][j]-full[1][i][j]) for i in range(3) for j in range(3)) < 1e-13
        assert max(abs(half[0][i][j]-half[1][i][j]) for i in range(3) for j in range(3)) > 1e-8
        for sign, complete, intermediate in zip([1, -1], full, half):
            squared = matrix_product(intermediate, intermediate)
            for time, kernel in [(interval, complete), (interval/2, intermediate)]:
                for i in range(3):
                    assert abs(sum(kernel[i])-1) < 1e-13 and min(kernel[i]) > 0
                    for j in range(3):
                        phase = sign*sqrt(3)*difference*time/2
                        closed = (1+2*exp(-1.5*rate_sum*time)*cos(phase-2*pi*((j-i)%3)/3))/3
                        assert abs(closed-kernel[i][j]) < 1e-13
                        assert abs(squared[i][j]-complete[i][j]) < 1e-13
        preparations += 1
    return {'aliasPreparations': preparations}


if __name__ == '__main__':
    print(json.dumps({'status': 'passed', **verify_path_information(), **verify_regeneration(), **verify_aliases()}))
