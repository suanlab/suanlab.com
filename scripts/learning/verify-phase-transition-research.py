"""Independent microstate, quadrature, branch and interface checks; no site imports."""
from itertools import product
from math import comb, exp, fsum, gamma, lgamma, log, pi, sqrt, tanh, cosh
import json


def law(n, k=1., b=0.):
    ms = [j/n for j in range(-n, n+1, 2)]
    logs = [lgamma(n+1)-lgamma(j+1)-lgamma(n-j+1)
            + n*(k*m*m/2+b*m) for j, m in enumerate(ms)]
    top = max(logs)
    weights = [exp(x-top) for x in logs]
    z = fsum(weights)
    return ms, [w/z for w in weights], logs


def simpson(f, lo, hi, count=20000):
    step = (hi-lo)/count
    return step/3*(f(lo)+f(hi)+fsum((4 if j % 2 else 2)*f(lo+j*step)
                                 for j in range(1, count)))


def positive_root(k):
    lo, hi = sqrt(1-1/k), 1.
    for _ in range(100):
        mid = (lo+hi)/2
        if tanh(k*mid) > mid:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def main():
    micro = 0
    for n in range(2, 13):
        for k, b in [(1., 0.), (2., .7/n)]:
            ms, ps, _ = law(n, k, b)
            bins = {j: [] for j in range(-n, n+1, 2)}
            for state in product([-1, 1], repeat=n):
                total = sum(state)
                bins[total].append(exp(k*total*total/(2*n)+b*total))
            z = fsum(fsum(v) for v in bins.values())
            for j, p in zip(bins, ps):
                assert len(bins[j]) == comb(n, (n+j)//2)
                assert abs(fsum(bins[j])/z-p) < 3e-14
            micro += 1
    moments = []
    norm = 12**.25*gamma(.25)/2
    for power in (0, 2, 4, 6, 8):
        measured = simpson(lambda x: x**power*exp(-x**4/12), -6, 6)/norm
        predicted = 12**(power/4)*gamma((power+1)/4)/gamma(.25)
        assert abs(measured-predicted) < 2e-11*max(1, predicted)
        moments.append(measured)
    c2 = sqrt(12)*gamma(.75)/gamma(.25)
    binder = 1-1/c2**2
    errors = []
    for n in (32, 128, 512, 2048, 8192):
        ms, ps, _ = law(n)
        e2 = fsum(p*(n**.25*m)**2 for m, p in zip(ms, ps))
        e4 = fsum(p*(n**.25*m)**4 for m, p in zip(ms, ps))
        assert abs(fsum(p*m for m, p in zip(ms, ps))) < 2e-14
        errors.append((abs(e2-c2), abs(1-e4/(3*e2**2)-binder)))
    assert all(a[0] > b[0] and a[1] > b[1] for a, b in zip(errors, errors[1:]))
    assert errors[-1][0] < .004 and errors[-1][1] < .004

    branches = 0
    for k in (1.5, 2., 2.5):
        root = positive_root(k)
        q = 1/(1-root*root)-k
        s = -(1+root)/2*log((1+root)/2)-(1-root)/2*log((1-root)/2)
        psi = -k*root*root/2-s
        log_a = -.5*log((1-root*root)*q)
        previous = None
        for n in (512, 2048, 8192):
            ms, ps, logs = law(n, k, 0.)
            selected = [v for m, v in zip(ms, logs) if m > 0]
            top = max(selected)
            log_z = top+log(fsum(exp(x-top) for x in selected))
            error = abs(exp(log_z+n*psi-log_a)-1)
            assert error < .02
            if previous is not None:
                assert error < previous
            previous = error
            branches += 1
        for v in (-1., 0., .7, 2.):
            n = 8192
            ms, ps, _ = law(n, k, v/n)
            mean = fsum(p*m for m, p in zip(ms, ps))
            var = fsum(p*m*m for m, p in zip(ms, ps))-mean**2
            plus = fsum(p*(1 if m > 0 else .5 if m == 0 else 0) for m, p in zip(ms, ps))
            assert abs(mean-root*tanh(v*root)) < .0006
            assert abs(plus-1/(1+exp(-2*v*root))) < .0004
            assert abs(var-root**2/cosh(v*root)**2) < .0006
            branches += 1

    interfaces = drives = 0
    for b, c, f0, kappa in [(-4., 3., 1., 1.), (-1., 1., 2., .5), (-2., 4., .3, 7.)]:
        a = 3*b*b/(16*c)
        m0 = sqrt(-3*b/(4*c))
        ell = sqrt(3*kappa/(f0*c))/(2*m0*m0)
        sigma = m0**4*sqrt(kappa*f0*c/3)/4
        numeric = simpson(lambda m: sqrt(kappa*f0*c/3)*m*(m0*m0-m*m), 0., m0)
        assert abs(numeric-sigma) < 2e-13*max(1, sigma)
        for x in (-7*ell, -ell, 0., ell, 7*ell):
            u = 1/(1+exp(-x/ell))
            m = m0*sqrt(u)
            d1 = m*(1-u)/(2*ell)
            d2 = m*(1-u)*(1-3*u)/(4*ell**2)
            phi = a*m*m/2+b*m**4/4+c*m**6/6
            assert abs(kappa*d1*d1/2-f0*phi) < 1e-12*max(1, sigma)
            assert abs(kappa*d2-f0*(a*m+b*m**3+c*m**5)) < 1e-12*max(1, sigma)
            interfaces += 1
        for fraction in (.0001, .0005, .001):
            eps = fraction*a
            y = (-b+sqrt(b*b-4*(a-eps)*c))/(2*c)
            delta = -f0*((a-eps)*y/2+b*y*y/4+c*y**3/6)
            assert delta > 0
            assert abs(delta/(f0*eps*m0*m0/2)-1) < .002
            radius = 2*sigma/delta
            barrier = 16*pi*sigma**3/(3*delta**2)
            energy = lambda r: 4*pi*sigma*r*r-4*pi*delta*r**3/3
            assert abs(energy(radius)/barrier-1) < 2e-14
            assert energy(.99*radius) < barrier and energy(1.01*radius) < barrier
            assert radius/ell > 100
            drives += 1
    return {'status': 'passed', 'microstatePreparations': micro,
            'quarticMomentIntegrals': len(moments), 'criticalFiniteSizes': len(errors),
            'laplaceAndWeakFieldPreparations': branches, 'interfaceEquationChecks': interfaces,
            'nearCoexistenceDroplets': drives}


if __name__ == '__main__':
    print(json.dumps(main(), indent=2))
