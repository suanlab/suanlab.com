"""Independent finite-model checks; standard library only, not a flow solver."""
from fractions import Fraction as F
from itertools import product
from math import sin, cos, pi, exp, expm1, log, isclose
import json

# Exact Fourier coefficients in Q(i), with averages taken by the zero mode.
ZERO = (F(0), F(0))


def plus(a, b):
    return a[0] + b[0], a[1] + b[1]


def times(a, b):
    return a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0]


def scale(poly, a):
    return {k: (a*v[0], a*v[1]) for k, v in poly.items()}


def add(*polys):
    out = {}
    for poly in polys:
        for k, v in poly.items():
            out[k] = plus(out.get(k, ZERO), v)
    return out


def mul(a, b):
    out = {}
    for k, v in a.items():
        for l, w in b.items():
            key = (k[0]+l[0], k[1]+l[1])
            out[key] = plus(out.get(key, ZERO), times(v, w))
    return out


def derivative(poly, axis):
    return {k: (-k[axis]*v[1], k[axis]*v[0]) for k, v in poly.items()}


def sine(m, n):
    return {(m, n): (F(0), -F(1, 2)), (-m, -n): (F(0), F(1, 2))}


def mean(poly):
    a, b = poly.get((0, 0), ZERO)
    assert b == 0
    return a


def check():
    counts = {}
    count = 0
    for mach, left, right in product([F(1, 10), F(3, 10), F(3, 5), F(9, 10)],
                                     [F(-1), F(-1, 2), F(0), F(1)],
                                     [F(-1), F(-1, 2), F(0), F(1)]):
        # eta_out^2 = 1 at each end. Work with squared raw amplitudes
        # to avoid assuming a floating square root is exact.
        wp2 = left**2 * 4/(1+mach)
        wm2 = 4/(1-mach)
        flux_left = ((1+mach)*wp2+(mach-1)*wm2)/4
        wm2_right = right**2 * 4/(1-mach)
        wp2_right = 4/(1+mach)
        flux_right = ((1+mach)*wp2_right+(mach-1)*wm2_right)/4
        assert flux_left == left**2-1 <= 0
        assert flux_right == 1-right**2 >= 0
        assert ((1+mach)+(mach-1))/4 == mach/2 > 0
        assert (1/(1+mach)+1/(1-mach)) == 2/(1-mach**2)
        count += 1
    counts['exactNormBoundaryCases'] = count

    # Fourier inner products independently compute all three printed integrals.
    count = positive = 0
    for a, b, amplitude in product([F(-2), F(1), F(3, 2)],
                                   [F(-1, 5), F(1, 3), F(2)],
                                   [F(-3), F(0), F(20)]):
        wx = scale(sine(2, 1), -b)
        wy = add(scale(sine(1, 0), a), scale(sine(2, 1), 2*b))
        div = add(derivative(wx, 0), derivative(wy, 1))
        assert all(v == ZERO for v in div.values())
        assert mean(wx) == mean(wy) == 0
        w2 = mean(add(mul(wx, wx), mul(wy, wy)))
        grad2 = mean(add(*(mul(derivative(w, j), derivative(w, j))
                           for w in [wx, wy] for j in [0, 1])))
        strain = scale(mul(sine(1, 0), sine(0, 1)), amplitude)
        exchange = mean(mul(strain, add(mul(wx, wx), scale(mul(wy, wy), -1))))
        assert w2 == (a*a+5*b*b)/2
        assert grad2 == (a*a+25*b*b)/2
        assert exchange == amplitude*a*b/2
        count += 1
    counts['exactTwoModeFourierBudgets'] = count
    for r in [F(1, 2), F(1), F(5), F(10), F(20)]:
        a, b = F(1), F(-1, 5)
        budget = -r*a*b/2-(a*a+25*b*b)/2
        assert budget == r/10-1
        assert (budget > 0) == (r > 10)
        positive += 1
    counts['exactInitialGrowthSignCases'] = positive

    # Independent composite midpoint integration of the stated differential
    # bound, rather than evaluating its closed primitive twice.
    count = 0
    for r, z in product([.5, 1., 5., 20.], [.01, .3, 2., 5.]):
        n = 20000
        integral = sum((r*exp(-(j+.5)*z/n)-1)*z/n for j in range(n))
        closed = r*(-expm1(-z))-z
        assert isclose(integral, closed, abs_tol=2e-7, rel_tol=2e-8)
        count += 1
    counts['independentEnvelopeIntegrations'] = count

    # A compact contact profile with zero inflow. The translated profile has
    # the same q=0 but loses full-state norm only through the right boundary.
    count = 0
    n = 16000
    for travel in [0., .1, .4, .7, .95, 1.]:
        upper = max(0., 1-travel)
        numeric = sum(sin(pi*(j+.5)*upper/n)**4*upper/n for j in range(n))
        primitive = 3*upper/8-sin(2*pi*upper)/(4*pi)+sin(4*pi*upper)/(32*pi)
        assert isclose(numeric, primitive, abs_tol=2e-9)
        if travel == 0:
            assert isclose(numeric, 3/8, abs_tol=1e-12)
        # d/d(travel) integral = -h(right)^2 / h0^2
        if 0 < travel < 1:
            eps = 1e-6
            def stored(s):
                y = 1-s
                return 3*y/8-sin(2*pi*y)/(4*pi)+sin(4*pi*y)/(32*pi)
            assert isclose((stored(travel+eps)-stored(travel-eps))/(2*eps),
                           -sin(pi*(1-travel))**4, abs_tol=1e-9)
        count += 1
    counts['independentContactStorageAndExitCases'] = count

    # Exact zero-base constant solutions defeat an unrestricted decay claim.
    count = 0
    for bx, by in product([F(-2), F(0), F(3, 2)], repeat=2):
        norm = (bx*bx+by*by)/2
        assert norm >= 0
        assert (norm == 0) == (bx == by == 0)
        constant = [{(0, 0): (v, F(0))} for v in [bx, by]]
        assert all(coef == ZERO for field in constant for axis in [0, 1]
                   for coef in derivative(field, axis).values())
        assert mean(add(*(mul(field, field) for field in constant)))/2 == norm
        count += 1
    counts['exactPersistentMeanErrorCases'] = count

    # Verify the nonconstant shifted solution by numerical derivatives of its
    # actual x-b*t arguments, including the shifted pressure.
    count = 0
    for t, bx, by in product([0., .2, .7], [-.4, .3], [-.2, .5]):
        nu, k, U, rho = .12, 1.3, .8, 1.7
        x, y, eps = .27, -.31, 1e-5
        def fields(x, y, t):
            X, Y = k*(x-bx*t), k*(y-by*t)
            A = U*exp(-2*nu*k*k*t)
            return (-A*cos(X)*sin(Y)+bx, A*sin(X)*cos(Y)+by,
                    -rho*A*A/4*(cos(2*X)+cos(2*Y)))
        f = fields(x, y, t)
        xp, xm = fields(x+eps,y,t), fields(x-eps,y,t)
        yp, ym = fields(x,y+eps,t), fields(x,y-eps,t)
        tp, tm = fields(x,y,t+eps), fields(x,y,t-eps)
        for j in [0, 1]:
            dt = (tp[j]-tm[j])/(2*eps)
            dx, dy = (xp[j]-xm[j])/(2*eps), (yp[j]-ym[j])/(2*eps)
            dp = ((xp[2]-xm[2]) if j == 0 else (yp[2]-ym[2]))/(2*eps)
            lap = (xp[j]+xm[j]+yp[j]+ym[j]-4*f[j])/(eps*eps)
            assert abs(dt+f[0]*dx+f[1]*dy+dp/rho-nu*lap) < 3e-6
        count += 1
    counts['independentShiftedSolutionMomentumCases'] = count
    return counts


if __name__ == '__main__':
    print(json.dumps(check(), sort_keys=True))
