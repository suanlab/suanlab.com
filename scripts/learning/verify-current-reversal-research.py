"""Independent Gaussian block-KL checks and exact ring fluctuation identities.

Numerical Gaussian and Poisson comparisons use binary floating point.
Fraction identities are exact for the specified finite ring, not general networks.
"""
from fractions import Fraction as F
from math import comb, cos, exp, expm1, isclose, log, sin, sqrt
import json


def inverse_and_determinant(matrix):
    n = len(matrix)
    work = [list(row)+[float(i == j) for j in range(n)] for i, row in enumerate(matrix)]
    determinant = 1.
    for k in range(n):
        pivot = max(range(k, n), key=lambda i: abs(work[i][k]))
        if pivot != k:
            work[k], work[pivot] = work[pivot], work[k]
            determinant = -determinant
        divisor = work[k][k]
        assert abs(divisor) > 1e-14
        determinant *= divisor
        work[k] = [x/divisor for x in work[k]]
        for i in range(n):
            if i != k:
                c = work[i][k]
                work[i] = [x-c*y for x, y in zip(work[i], work[k])]
    return [row[n:] for row in work], determinant


def poisson_weights(mean, cutoff):
    weights = [exp(-mean)]
    for n in range(1, cutoff+1):
        weights.append(weights[-1]*mean/n)
    assert cutoff+2 > mean
    bound = weights[-1]*mean/(cutoff+1)/(1-mean/(cutoff+2))
    return weights, bound


def negative_probability(a, b, duration, cutoff=24):
    p, bp = poisson_weights(a*duration, cutoff)
    m, bm = poisson_weights(b*duration, cutoff)
    sign = (a > b)-(a < b)
    value = sum(pi*mj for i, pi in enumerate(p) for j, mj in enumerate(m) if (i-j)*sign < 0)
    return value, bp+bm


def verify():
    gaussian, max_kl_error = 0, 0.
    for gamma in [.25, .5, 1., 2.]:
        for diffusion in [.25, 1., 2.]:
            for omega in [-3., -1., 0., .5, 2.]:
                for delta in [.05, .25, 1., 3.]:
                    v, a, angle = diffusion/gamma, exp(-gamma*delta), omega*delta
                    c, s = a*cos(angle), a*sin(angle)
                    mat = [[c, -s], [s, c]]
                    # Joint covariance in order (X_j, X_{j+1}); reverse swaps the blocks.
                    cov = [[v if i == j else 0. for j in range(4)] for i in range(4)]
                    for i in range(2):
                        for j in range(2):
                            cov[i+2][j] = cov[j][i+2] = v*mat[i][j]
                    permutation = [2, 3, 0, 1]
                    backward = [[cov[permutation[i]][permutation[j]] for j in range(4)] for i in range(4)]
                    inv, det_back = inverse_and_determinant(backward)
                    _, det_forward = inverse_and_determinant(cov)
                    trace = sum(inv[i][j]*cov[j][i] for i in range(4) for j in range(4))
                    block_kl = .5*(trace-4+log(det_back/det_forward))
                    expected = 4*sin(angle)**2/expm1(2*gamma*delta)
                    max_kl_error = max(max_kl_error, abs(block_kl-expected))
                    assert isclose(block_kl, expected, rel_tol=2e-11, abs_tol=2e-12)
                    assert 0 <= expected/delta <= 2*omega**2/gamma+1e-12
                    q = -v*expm1(-2*gamma*delta)
                    for x, y in [((1., 0.), (0., 1.)), ((2., -1.), (-.5, 3.)), ((0., 0.), (2., 1.))]:
                        def residual(u, z):
                            return sum((z[i]-sum(mat[i][j]*u[j] for j in range(2)))**2 for i in range(2))
                        expanded = (sum(t*t for t in y)-sum(t*t for t in x))/(2*v)+(residual(y, x)-residual(x, y))/(2*q)
                        area_value = 2*a*sin(angle)/q*(x[0]*y[1]-x[1]*y[0])
                        assert isclose(expanded, area_value, rel_tol=2e-11, abs_tol=2e-11)
                    gaussian += 1

    pairs = [(F(3), F(1)), (F(2), F(1)), (F(1), F(3)), (F(1), F(1)), (F(1, 2), F(2))]
    exact = 0
    for a, b in pairs:
        p = a/(a+b)
        ratio = a/b
        for n in range(25):
            probabilities = [F(comb(n, j))*p**j*(1-p)**(n-j) for j in range(n+1)]
            assert sum(probabilities) == 1
            assert sum(probabilities[j]*ratio**(-(2*j-n)) for j in range(n+1)) == 1
            mean = sum((2*j-n)*probabilities[j] for j in range(n+1))
            variance = sum((2*j-n-mean)**2*probabilities[j] for j in range(n+1))
            assert mean == n*(a-b)/(a+b)
            assert variance == 4*n*a*b/(a+b)**2
            for j in range(n+1):
                assert probabilities[j] == ratio**(2*j-n)*probabilities[n-j]
            exact += 1

    poisson_cases, max_probability_difference = 0, 0.
    for fa, fb in pairs:
        a, b = float(fa), float(fb)
        for duration in [0., .1, .5, 1., 2.]:
            observed, omitted = negative_probability(a, b, duration)
            total, tail = poisson_weights((a+b)*duration, 80)
            p = a/(a+b)
            sign = (a > b)-(a < b)
            # Separate oracle: condition on the total count, then split binomially.
            oracle = sum(w*comb(n, j)*p**j*(1-p)**(n-j)
                         for n, w in enumerate(total) for j in range(n+1)
                         if (2*j-n)*sign < 0)
            assert observed <= oracle+2e-14
            assert oracle-observed <= omitted+tail+2e-14
            max_probability_difference = max(max_probability_difference, abs(oracle-observed))
            L = log(a/b)
            for s in [-1., 0., .25, .5, 1., 2.]:
                psi = a*expm1(-s*L)+b*expm1(s*L)
                reflected = a*expm1(-(1-s)*L)+b*expm1((1-s)*L)
                assert isclose(psi, reflected, rel_tol=1e-12, abs_tol=1e-12)
            poisson_cases += 1
    return {'status': 'passed', 'gaussianBlockPreparations': gaussian,
            'maxGaussianKlDifference': max_kl_error, 'exactConditionalCountPreparations': exact,
            'poissonOraclePreparations': poisson_cases,
            'maxTruncatedNegativeProbabilityDifference': max_probability_difference,
            'scope': 'Gaussian block KL and positive count-series comparisons; exact Fraction identities at fixed total count. No empirical confidence or general thermal interpretation.'}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
