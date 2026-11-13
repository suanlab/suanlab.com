"""Independent current-corrector, initial-work and marked-reservoir references.

Fraction checks are exact. Matrix exponential comparisons use floating point.
The zero-attenuation work cases explicitly represent the fully relaxed limit.
"""
from pathlib import Path
from fractions import Fraction as F
from itertools import product as cartesian
from math import exp, expm1, log, sqrt
import importlib.util
import json
import sys
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('response_reference', Path(__file__).with_name('verify-response-research.py'))
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
solve, matrix_exp, multiply = reference.solve, reference.matrix_exp, reference.product


def determinant3(a):
    return (a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
            - a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
            + a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]))


def verify():
    current_cases, polynomial_cases, max_polynomial_error = 0, 0, 0.
    increment = [[F(0), F(1), F(-1)], [F(-1), F(0), F(1)], [F(1), F(-1), F(0)]]
    worked_rates = [(3, 3, 3, 1, 1, 1), (2, 3, 4, 1, 1, 1), (1, 2, 5, 2, 1, 1), (1, 1, 1, 3, 3, 3)]
    for index, rates in enumerate([*cartesian([1, 2, 4], repeat=6), *worked_rates]):
        a, b, c, d, e, f = map(F, rates)
        q = [[-a-f, a, f], [d, -d-b, b], [c, e, -c-e]]
        tree = [c*d+d*e+b*c, a*e+a*c+e*f, b*f+a*b+d*f]
        z = sum(tree)
        stationary = [w/z for w in tree]
        drift = [sum(q[i][j]*increment[i][j] for j in range(3) if i != j) for i in range(3)]
        mean = sum(p*v for p, v in zip(stationary, drift))
        h = solve(q[:2]+[stationary], [mean-drift[0], mean-drift[1], 0])
        assert sum(p*v for p, v in zip(stationary, h)) == 0
        assert [sum(q[i][j]*h[j] for j in range(3)) for i in range(3)] == [mean-v for v in drift]
        variance = sum(stationary[i]*q[i][j]*(increment[i][j]+h[j]-h[i])**2 for i in range(3) for j in range(3) if i != j)
        assert mean == 3*(a*b*c-d*e*f)/z
        expected = (9*(a*b*c+d*e*f)-2*sum(rates)*mean**2)/z
        assert variance == expected and variance > 0
        if mean:
            sigma = float(mean/3)*log(float(a*b*c/(d*e*f)))
            assert float(variance)*sigma/float(mean)**2 >= 2-1e-12
        if a == b == c and d == e == f:
            assert mean == a-d and variance == a+d
        if index % 9 == 0 or rates in worked_rates:
            for chi in [-.3, .2, .5]:
                root = .37
                shifted = [[(root if i == j else 0.)-float(q[i][j])*(exp(chi*float(increment[i][j])) if i != j else 1.) for j in range(3)] for i in range(3)]
                direct = determinant3(shifted)
                polynomial = root**3+sum(rates)*root**2+float(z)*root+float(a*b*c)*(1-exp(3*chi))+float(d*e*f)*(1-exp(-3*chi))
                error = abs(direct-polynomial)
                max_polynomial_error = max(max_polynomial_error, error)
                assert error < 1e-9
                polynomial_cases += 1
        current_cases += 1

    work_cases, reweightings = 0, 0
    values = [F(1, 3), F(1), F(2), F(3)]
    for r0, r1, r2, attenuation in cartesian(values, values, values, [F(0), F(1, 4), F(1, 2), F(1)]):
        pi0 = [r0/(1+r0), 1/(1+r0)]
        hold = [r1/(1+r1), 1/(1+r1)]
        final = [r2/(1+r2), 1/(1+r2)]
        p1 = [[attenuation*int(x == y)+(1-attenuation)*hold[y] for y in range(2)] for x in range(2)]
        normalizer = (1+1/r0)/(1+1/r2)
        y = [[normalizer*(r0/r1)**x*(r1/r2)**z for z in range(2)] for x in range(2)]
        assert sum(pi0[x]*p1[x][z]*y[x][z] for x in range(2) for z in range(2)) == 1
        c = [pi0[x]**2*sum(p1[x][z]*y[x][z]**2 for z in range(2)) for x in range(2)]
        minimum = (sqrt(float(c[0]))+sqrt(float(c[1])))**2-1
        optimal = [sqrt(float(v))/sum(sqrt(float(w)) for w in c) for v in c]
        assert all(0 < p < 1 for p in optimal)
        assert abs(sum(float(v)/p for v, p in zip(c, optimal))-1-minimum) < 1e-12
        for initial_one in [F(1, 10), F(1, 2), F(9, 10)]:
            p = [1-initial_one, initial_one]
            corrected = sum(p[x]*p1[x][z]*(pi0[x]/p[x])*y[x][z] for x in range(2) for z in range(2))
            second = sum(p[x]*p1[x][z]*((pi0[x]/p[x])*y[x][z])**2 for x in range(2) for z in range(2))
            assert corrected == 1 and second == sum(v/w for v, w in zip(c, p))
            assert minimum <= float(second-1)+1e-12
            reweightings += 1
        if attenuation == 1:
            assert c == [v*v for v in final]
        if (r0, r1, r2) == (F(1), F(3), F(2)):
            second_moments = [sum(p1[x][z]*y[x][z]**2 for z in range(2)) for x in range(2)]
            assert second_moments == [(21-5*attenuation)/9, (7+5*attenuation)/27]
        work_cases += 1

    # Match the manuscript's finite one-second hold with a separate matrix exponential.
    transition = matrix_exp([[-.25, .25], [.75, -.75]], 1.)
    weights = [[4/3, 2.], [4/9, 2/3]]
    attenuation = exp(-1.)
    second_by_initial = [sum(transition[x][y]*weights[x][y]**2 for y in range(2)) for x in range(2)]
    assert abs(second_by_initial[0]-(21-5*attenuation)/9) < 1e-12
    assert abs(second_by_initial[1]-(7+5*attenuation)/27) < 1e-12
    c = [.25*v for v in second_by_initial]
    optimum = sqrt(c[1])/(sqrt(c[0])+sqrt(c[1]))
    physical_proposals = 0
    for initial_one in [.2, .5, .8, optimum]:
        initial = [1-initial_one, initial_one]
        corrected = sum(initial[x]*transition[x][y]*(.5/initial[x])*weights[x][y] for x in range(2) for y in range(2))
        raw = sum(initial[x]*transition[x][y]*weights[x][y] for x in range(2) for y in range(2))
        second = sum(initial[x]*transition[x][y]*((.5/initial[x])*weights[x][y])**2 for x in range(2) for y in range(2))
        assert abs(corrected-1) < 1e-12
        assert abs(raw-((1-initial_one)*(9-attenuation)+initial_one*(3+attenuation))/6) < 1e-12
        assert abs(second-sum(v/w for v, w in zip(c, initial))) < 1e-12
        physical_proposals += 1

    reservoir_cases, finite_times, max_finite_variance_error, max_mean_error = 0, 0, 0., 0.
    for index, (r, ba, bb, na, nb) in enumerate(cartesian([F(1, 3), F(1), F(2), F(3), F(4)], [1, 2, 3], [1, 2, 3], [F(1, 2), F(1), F(2)], [F(1, 2), F(1), F(2)])):
        aa, da = na/(1+r**ba), na*r**ba/(1+r**ba)
        ab, db = nb/(1+r**bb), nb*r**bb/(1+r**bb)
        a, d = aa+ab, da+db
        k = a+d
        stationary = [d/k, a/k]
        cp, cm = aa*db, ab*da
        current = (cp-cm)/k
        variance = (cp+cm)/k-2*(cp-cm)**2/k**3
        activity = stationary[0]*aa+stationary[1]*da
        assert current == stationary[0]*aa-stationary[1]*da
        delta_h = (current-aa)/a
        bracket = stationary[0]*(aa*(1+delta_h)**2+ab*delta_h**2)+stationary[1]*(da*(-1-delta_h)**2+db*delta_h**2)
        assert variance == bracket and variance > 0 and activity > variance
        affinity = log(float(cp/cm))
        sigma = float(current)*affinity
        assert sigma >= -1e-13
        if current:
            assert sigma+1e-12 >= float(2*current**2/variance)
        # The endpoint-conjugated heat tilt is exactly the aggregate generator.
        heat_up, heat_down = aa*cm/cp+ab, da*cp/cm+db
        endpoint = stationary[0]/(stationary[1]*r**bb)
        assert heat_up/endpoint == a and heat_down*endpoint == d
        worked = (r, ba, bb, na, nb) == (F(3), 1, 2, F(1), F(1))
        if index % 7 == 0 or worked:
            q = [[float(-a), float(a)], [float(d), float(-d)]]
            b1, b2 = [[0., float(aa)], [-float(da), 0.]], [[0., float(aa)], [float(da), 0.]]
            block = [[0.]*6 for _ in range(6)]
            for i in range(2):
                for j in range(2):
                    for offset in [0, 2, 4]:
                        block[offset+i][offset+j] = q[i][j]
                    block[i][2+j], block[i][4+j], block[2+i][4+j] = b1[i][j], b2[i][j], 2*b1[i][j]
            for duration in [0., .1, .5, 1., 2., 3.]:
                moments = multiply([[float(stationary[0]), float(stationary[1]), 0., 0., 0., 0.]], matrix_exp(block, duration))[0]
                mean = sum(moments[2:4])
                var = sum(moments[4:6])-mean**2
                expected = float(variance)*duration+float((activity-variance)/k)*(-expm1(-float(k)*duration))
                max_mean_error = max(max_mean_error, abs(mean-float(current)*duration))
                max_finite_variance_error = max(max_finite_variance_error, abs(var-expected))
                assert abs(mean-float(current)*duration) < 1e-10 and abs(var-expected) < 1e-10
                finite_times += 1
        if (r, ba, bb, na, nb) == (F(3), 1, 2, F(1), F(1)):
            assert current == F(3, 40) and variance == F(231, 1600) and activity == F(27, 80)
            tilted = [[float(-a), float(heat_up)], [float(heat_down), float(-d)]]
            for duration in [.1, .5, 1., 2.]:
                moment = sum(multiply([list(map(float, stationary))], matrix_exp(tilted, duration))[0])
                assert abs(moment-(1-expm1(-2*duration)/16)) < 1e-12
        reservoir_cases += 1
    return {'status': 'passed', 'exactCurrentPreparations': current_cases,
            'tiltedPolynomialPreparations': polynomial_cases, 'maxPolynomialDifference': max_polynomial_error,
            'exactWorkPreparations': work_cases, 'exactInitialReweightings': reweightings,
            'workPhysicalProposalPreparations': physical_proposals, 'workIncludesFullyRelaxedLimit': True, 'exactReservoirPreparations': reservoir_cases,
            'finiteCountMomentPreparations': finite_times, 'maxFiniteVarianceDifference': max_finite_variance_error,
            'maxFiniteMeanDifference': max_mean_error,
            'scope': 'Exact current-corrector, work reweighting and marked-channel identities; floating determinant and finite-time moment comparisons. No empirical confidence or directed-rounding certificate.'}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
