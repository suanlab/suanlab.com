"""Independent spectral transforms, exact handle algebra and path-score moments.

Floating matrix exponentials and quadrature are prototype comparisons, not
interval certificates. Fraction calculations are exact for the stated rates.
"""
from fractions import Fraction as F
from math import ceil, cos, exp, log2, pi, sin
import json


def product(a, b):
    return [[sum(x*y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def matrix_exp(a, t):
    n = len(a)
    norm = max(sum(abs(v) for v in row) for row in a)*abs(t)
    scaling = max(0, ceil(log2(norm))) if norm else 0
    b = [[v*t/2**scaling for v in row] for row in a]
    term = [[float(i == j) for j in range(n)] for i in range(n)]
    total = [row[:] for row in term]
    for k in range(1, 100):
        term = [[v/k for v in row] for row in product(term, b)]
        total = [[x+y for x, y in zip(r, q)] for r, q in zip(total, term)]
        if max(sum(abs(v) for v in row) for row in term) < 2e-17:
            break
    else:
        raise AssertionError('Taylor series did not converge')
    for _ in range(scaling):
        total = product(total, total)
    return total


def solve(a, rhs):
    n = len(rhs)
    w = [[F(x) for x in row]+[F(rhs[i])] for i, row in enumerate(a)]
    for k in range(n):
        p = next(i for i in range(k, n) if w[i][k])
        w[p], w[k] = w[k], w[p]
        divisor = w[k][k]
        w[k] = [x/divisor for x in w[k]]
        for i in range(n):
            if i != k:
                v = w[i][k]
                w[i] = [x-v*y for x, y in zip(w[i], w[k])]
    return [row[-1] for row in w]


def rate_system(rates, eta):
    a, b, c, d, e, f = map(F, rates)
    q = [[-a-f, a, f], [d, -d-b, b], [c, e, -c-e]]
    tree = [c*d+d*e+b*c, a*e+a*c+e*f, b*f+a*b+d*f]
    stationary = [x/sum(tree) for x in tree]
    g = [[F(0)]*3 for _ in range(3)]
    g[0][1], g[1][0] = eta+F(1, 2), eta-F(1, 2)
    q1 = [[q[i][j]*g[i][j] if i != j else F(0) for j in range(3)] for i in range(3)]
    k2 = [[q[i][j]*g[i][j]**2 if i != j else F(0) for j in range(3)] for i in range(3)]
    for i in range(3):
        q1[i][i] = -sum(q1[i])
    return q, stationary, g, q1, k2


def verify():
    spectra, max_spectrum_error, max_probe_error = 0, 0., 0.
    for mu in [.5, 1., 2.]:
        for stiffness in [.5, 1., 2.]:
            for theta in [.25, 1., 3.]:
                gamma, diffusion = mu*stiffness, mu*theta
                for omega in [-2., 0., 1., 3.]:
                    for frequency in [.125, .5, 1., 3.]:
                        z = gamma+1j*frequency
                        det = z*z+omega*omega
                        h = [[z/det, -omega/det], [omega/det, z/det]]
                        dagger = [[h[j][i].conjugate() for j in range(2)] for i in range(2)]
                        s = [[2*diffusion*v for v in row] for row in product(h, dagger)]
                        expected = diffusion/(gamma**2+(frequency-omega)**2)+diffusion/(gamma**2+(frequency+omega)**2)
                        cross = 4j*diffusion*omega*frequency/abs(det)**2
                        max_spectrum_error = max(max_spectrum_error, abs(s[0][0]-expected), abs(s[0][1]-cross))
                        assert abs(s[0][0]-expected) < 1e-11 and abs(s[0][1]-cross) < 1e-11
                        assert abs(s[1][0]-s[0][1].conjugate()) < 1e-12
                        assert s[0][0].real > 0 and (s[0][0]*s[1][1]-s[0][1]*s[1][0]).real > -1e-11
                        chi = mu*h[0][0]
                        defect = expected+2*theta*chi.imag/frequency
                        assert abs(defect-4*diffusion*omega**2/abs(det)**2) < 1e-11
                        # Average deterministic probe work over a full period separately.
                        average = sum(cos(2*pi*k/128)*(-frequency*chi.real*sin(2*pi*k/128)-frequency*chi.imag*cos(2*pi*k/128)) for k in range(128))/128
                        exact_power = -frequency*chi.imag/2
                        max_probe_error = max(max_probe_error, abs(average-exact_power))
                        assert abs(average-exact_power) < 1e-12
                        spectra += 1

    quadrature, max_transform_difference = 0, 0.
    for gamma in [.25, 1., 4.]:
        for omega in [-3., 0., 3.]:
            for frequency in [.25, 1., 3.]:
                # v=1, D=gamma. Direct two-sided covariance Fourier integral.
                end, n = 32/gamma, 16384
                step = end/n
                total, cross = 0., 0.
                for k in range(n+1):
                    t = k*step
                    weight = 1 if k in (0, n) else 4 if k % 2 else 2
                    decay = exp(-gamma*t)
                    total += weight*decay*cos(omega*t)*cos(frequency*t)
                    cross += weight*decay*sin(omega*t)*sin(frequency*t)
                computed = 2*step/3*total
                cross_computed = 2j*step/3*cross
                z = gamma+1j*frequency
                expected = gamma/(gamma**2+(frequency-omega)**2)+gamma/(gamma**2+(frequency+omega)**2)
                expected_cross = 4j*gamma*omega*frequency/abs(z*z+omega**2)**2
                error = max(abs(computed-expected), abs(cross_computed-expected_cross))
                max_transform_difference = max(max_transform_difference, error)
                assert error < 2e-7
                quadrature += 1

    families = [(1, 1, 1, 1, 1, 1), (2, 3, 4, 1, 1, 1),
                (1, 1, 1, 2, 3, 4), (2, 3, 1, 1, 2, 3), (1, 4, 2, 3, 2, 1)]
    exact_cases, moment_cases, max_fisher_error, max_response_error = 0, 0, 0., 0.
    for rates in families:
        q, stationary, _, _, _ = rate_system(rates, F(0))
        fp, fm = stationary[0]*rates[0], stationary[1]*rates[3]
        traffic, current = fp+fm, fp-fm
        eta_star = -current/(2*traffic)
        for eta in sorted(set([F(-1), F(-1, 3), F(0), F(2, 3), F(1), eta_star])):
            q, stationary, g, q1, k2 = rate_system(rates, eta)
            source = product([stationary], q1)[0]
            info_rate = sum(product([stationary], k2)[0])
            assert sum(source) == 0
            assert info_rate == fp*(eta+F(1, 2))**2+fm*(eta-F(1, 2))**2
            assert info_rate-fp*fm/traffic == traffic*(eta-eta_star)**2
            derivative = solve([[q[i][j] for i in range(3)] for j in range(2)]+[[1, 1, 1]], [-source[0], -source[1], 0])
            assert product([derivative], q)[0] == [-x for x in source]
            if rates == (2, 3, 4, 1, 1, 1):
                assert derivative[1] == (135+138*eta)/1444
                assert eta_star == F(-23, 90)
            exact_cases += 1
            block = [[0.]*9 for _ in range(9)]
            for i in range(3):
                for j in range(3):
                    for offset in [0, 3, 6]:
                        block[offset+i][offset+j] = float(q[i][j])
                    block[i][3+j] = float(q1[i][j])
                    block[i][6+j] = float(k2[i][j])
                    block[3+i][6+j] = 2*float(q1[i][j])
            for duration in [.125, .5, 1., 2.]:
                moments = product([[*map(float, stationary), 0., 0., 0., 0., 0., 0.]], matrix_exp(block, duration))[0]
                first, second = moments[3:6], moments[6:9]
                fisher = float(info_rate)*duration
                max_fisher_error = max(max_fisher_error, abs(sum(second)-fisher))
                assert abs(sum(first)) < 1e-11 and abs(sum(second)-fisher) < 1e-10
                assert min(second) >= -1e-11
                assert first[1]**2 <= float(stationary[1]*(1-stationary[1]))*fisher+1e-11
                eps = .0005
                def perturbed(h):
                    a = [[float(q[i][j])*exp(h*float(g[i][j])) if i != j else 0. for j in range(3)] for i in range(3)]
                    for i in range(3):
                        a[i][i] = -sum(a[i])
                    return product([list(map(float, stationary))], matrix_exp(a, duration))[0][1]
                centered = (perturbed(eps)-perturbed(-eps))/(2*eps)
                max_response_error = max(max_response_error, abs(centered-first[1]))
                assert abs(centered-first[1]) < 1e-6
                # Scalar score transform uses a different diagonal from Q(h).
                z = .0002
                def transform(z):
                    a = [[float(q[i][j])*exp(z*float(g[i][j])) if i != j else float(q[i][i])-z*sum(float(q[i][k]*g[i][k]) for k in range(3) if k != i) for j in range(3)] for i in range(3)]
                    return sum(product([list(map(float, stationary))], matrix_exp(a, duration))[0])
                second_difference = (transform(z)+transform(-z)-2)/(z*z)
                assert abs(second_difference-fisher) < 3e-5
                moment_cases += 1
    return {'status': 'passed', 'spectralPreparations': spectra,
            'timeCovarianceQuadratures': quadrature, 'maxSpectralDifference': max_spectrum_error,
            'maxProbePeriodDifference': max_probe_error, 'maxFourierQuadratureDifference': max_transform_difference,
            'exactHandlePreparations': exact_cases, 'scoreMomentPreparations': moment_cases,
            'maxFisherMomentDifference': max_fisher_error, 'maxCenteredResponseDifference': max_response_error,
            'scope': 'Exact rational static/control identities; floating spectra, quadrature, score-tilt and finite-perturbation checks. No directed interval or empirical precision certificate.'}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
