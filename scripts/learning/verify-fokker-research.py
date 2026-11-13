"""Independent finite first-passage solves and OU transition consistency checks.

Exact rational grid solves do not certify continuum convergence with drift.
OU checks use binary floating point, not directed-rounding enclosures.
"""
from fractions import Fraction as F
from math import exp, expm1, isclose, log
import json


def solve(matrix, rhs):
    n = len(rhs)
    a = [[F(x) for x in row] + [F(rhs[i])] for i, row in enumerate(matrix)]
    for k in range(n):
        pivot = next(i for i in range(k, n) if a[i][k])
        a[k], a[pivot] = a[pivot], a[k]
        divisor = a[k][k]
        a[k] = [x / divisor for x in a[k]]
        for i in range(n):
            if i != k:
                v = a[i][k]
                a[i] = [x - v*y for x, y in zip(a[i], a[k])]
    return [row[-1] for row in a]


def brownian_system(cells, length, diffusion):
    dx = length / cells
    rate = diffusion / dx**2
    minus_a = [[F(0) for _ in range(cells)] for _ in range(cells)]
    left, right = [F(0)]*cells, [F(0)]*cells
    for i in range(cells):
        for neighbor in (i-1, i+1):
            if 0 <= neighbor < cells:
                minus_a[i][neighbor] = -rate
                minus_a[i][i] += rate
        if i == 0:
            left[i] = 2*rate
            minus_a[i][i] += left[i]
        if i == cells-1:
            right[i] = 2*rate
            minus_a[i][i] += right[i]
    return minus_a, left, right


def verify():
    grid_count = 0
    for cells in range(2, 25):
        for length in map(F, [1, 2, 4]):
            for diffusion in [F(1, 4), F(1, 2), F(1), F(2)]:
                mat, left, right = brownian_system(cells, length, diffusion)
                tau = solve(mat, [1]*cells)
                ur, ul = solve(mat, right), solve(mat, left)
                wr, wl = solve(mat, ur), solve(mat, ul)
                dx = length / cells
                for i in range(cells):
                    x = (F(i)+F(1, 2))*dx
                    assert tau[i] == x*(length-x)/(2*diffusion)+dx**2/(8*diffusion)
                    assert ur[i] == x/length
                    assert ul[i]+ur[i] == 1
                    assert wl[i]+wr[i] == tau[i]
                    assert 0 < ur[i] < 1 and tau[i] > 0
                grid_count += 1
    mat, _, right = brownian_system(2, F(2), F(1))
    ur = solve(mat, right)
    wr = solve(mat, ur)
    assert ur == [F(1, 4), F(3, 4)]
    assert wr == [F(3, 16), F(5, 16)]
    assert wr[0]/ur[0] == F(3, 4)
    assert wr[1]/ur[1] == F(5, 12)

    transitions = 0
    max_error = 0.0
    for gamma in [.05, .5, 1., 2., 5.]:
        for diffusion in [.125, .5, 1., 2.]:
            for delta in [.001, .1, .5, 1., 2.]:
                a = exp(-gamma*delta)
                q = -diffusion/gamma*expm1(-2*gamma*delta)
                recovered_g = -log(a)/delta
                recovered_d = recovered_g*q/(1-a*a)
                assert isclose(gamma, recovered_g, rel_tol=3e-12)
                assert isclose(diffusion, recovered_d, rel_tol=3e-12)
                # Compose conditional Gaussian kernels without using a whole-time formula.
                mean, variance = 2., .25
                for _ in range(7):
                    mean, variance = a*mean, a*a*variance+q
                t = 7*delta
                expected_v = .25*exp(-2*gamma*t)-diffusion/gamma*expm1(-2*gamma*t)
                max_error = max(max_error, abs(variance-expected_v))
                assert isclose(mean, 2*exp(-gamma*t), rel_tol=1e-12)
                assert isclose(variance, expected_v, rel_tol=1e-12, abs_tol=1e-12)
                assert 0 < (1-a)/delta < gamma
                assert 0 < q/(2*delta) < diffusion
                # Population covariance: independent observation noise affects lag zero only.
                c, noise = diffusion/gamma, .5
                r0, r1, r2 = c+noise, c*a, c*a*a
                assert isclose(r2/r1, a, rel_tol=1e-14)
                assert r1/r0 < a
                transitions += 1
    return {'status': 'passed', 'exactGridPreparations': grid_count,
            'ouTransitionPreparations': transitions,
            'maxComposedVarianceDifference': max_error,
            'scope': 'Exact rational uniform pure-diffusion grid solves; floating OU checks; no general continuum error certificate.'}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
