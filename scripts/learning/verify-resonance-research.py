"""Independent standard-library checks; no product or registry imports."""
import json
import math
import sys


def resonant_state(m, k, c, force, t):
    omega = math.sqrt(k / m)
    beta = c / (2 * m)
    d = math.sqrt(omega * omega - beta * beta)
    decay = math.exp(-beta * t)
    x = force / c * (math.sin(omega * t) / omega - decay * math.sin(d * t) / d)
    v = force / c * (math.cos(omega * t) - decay * (math.cos(d * t) - beta * math.sin(d * t) / d))
    a = force / c * (-omega * math.sin(omega * t) - decay * ((beta * beta - d * d) * math.sin(d * t) / d - 2 * beta * math.cos(d * t)))
    return x, v, a


def simpson(f, left, right, n=20000):
    step = (right - left) / n
    return step / 3 * (f(left) + f(right) + sum((4 if i % 2 else 2) * f(left + i * step) for i in range(1, n)))


def adaptive(f, left, right, tolerance=1e-10):
    def rule(a, b, fa, fm, fb):
        return (b - a) * (fa + 4 * fm + fb) / 6

    def visit(a, b, fa, fm, fb, whole, tol, depth):
        mid = (a + b) / 2
        fl, fr = f((a + mid) / 2), f((mid + b) / 2)
        low, high = rule(a, mid, fa, fl, fm), rule(mid, b, fm, fr, fb)
        delta = low + high - whole
        if abs(delta) <= 15 * tol:
            return low + high + delta / 15
        assert depth > 0, 'Quadrature budget exhausted'
        return visit(a, mid, fa, fl, fm, low, tol / 2, depth - 1) + visit(mid, b, fm, fr, fb, high, tol / 2, depth - 1)

    fa, fm, fb = f(left), f((left + right) / 2), f(right)
    return visit(left, right, fa, fm, fb, rule(left, right, fa, fm, fb), tolerance, 28)


def rk4(m, k, c, force, duration, steps):
    omega = math.sqrt(k / m)
    h = duration / steps
    x = v = 0.0
    def rhs(t, q, p):
        return p, (force * math.cos(omega * t) - c * p - k * q) / m
    for i in range(steps):
        t = i * h
        a, b = rhs(t, x, v)
        d, e = rhs(t + h / 2, x + h * a / 2, v + h * b / 2)
        f, g = rhs(t + h / 2, x + h * d / 2, v + h * e / 2)
        j, l = rhs(t + h, x + h * f, v + h * g)
        x += h * (a + 2 * d + 2 * f + j) / 6
        v += h * (b + 2 * e + 2 * g + l) / 6
    return x, v


def verify():
    counts = dict(initialValueJets=0, rk4Refinements=0, scaledLimitStates=0, transientBounds=0, spectralIntegrals=0, finiteWindowIntegrals=0, positivePowerStates=0, weakDampingFixedTimes=0, tailBounds=0)
    for m in [.5, 1, 2]:
        for k in [.5, 2]:
            omega = math.sqrt(k / m)
            for ratio in [.025, .2, .9]:
                beta, c = ratio * omega, 2 * m * ratio * omega
                for force in [.5, 2]:
                    assert all(abs(z) < 1e-12 for z in resonant_state(m, k, c, force, 0)[:2])
                    for t in [.1 / omega, 1 / omega, 8 / omega, 1 / beta]:
                        x, v, a = resonant_state(m, k, c, force, t)
                        assert abs(m * a + c * v + k * x - force * math.cos(omega * t)) < 2e-10
                        counts['initialValueJets'] += 1
                        scaled = c * c / (force * force) * (v * v + omega * omega * x * x)
                        d = math.sqrt(omega * omega - beta * beta)
                        bound = math.sqrt((1 + beta / d)**2 + (omega / d)**2)
                        assert abs(scaled - 1) <= 2 * bound * math.exp(-beta*t) + bound**2 * math.exp(-2*beta*t) + 1e-12
                        counts['transientBounds'] += 1
                    t = 3 / omega
                    expected = resonant_state(m, k, c, force, t)[:2]
                    coarse, fine = rk4(m, k, c, force, t, 600), rk4(m, k, c, force, t, 1200)
                    assert max(abs(x-y) for x,y in zip(coarse, fine)) < 2e-8
                    assert max(abs(x-y) for x,y in zip(fine, expected)) < 2e-9
                    counts['rk4Refinements'] += 1
            for tau in [.25, 1, 3]:
                for ratio in [.1, .01, .001]:
                    beta, c = ratio * omega, 2 * m * ratio * omega
                    x, v, _ = resonant_state(m, k, c, 1, tau/beta)
                    scaled = c*c*(v*v+omega*omega*x*x)
                    # Bound the vector difference without assuming convergence of the fast phase.
                    d = math.sqrt(omega*omega-beta*beta)
                    phase = beta*tau/(omega+d)
                    error = math.exp(-tau)*(2*phase+beta/d+omega/d-1)
                    target = 1-math.exp(-tau)
                    assert abs(scaled-target*target) <= 2*target*error+error*error+1e-11
                    counts['scaledLimitStates'] += 1
            c = 1e-6*m*omega
            t = 2/omega
            x, v, _ = resonant_state(m,k,c,1,t)
            undamped = (t*math.sin(omega*t)/(2*m*omega), (math.sin(omega*t)+omega*t*math.cos(omega*t))/(2*m*omega))
            assert max(abs(x-undamped[0]),abs(v-undamped[1])) < 2e-5
            counts['weakDampingFixedTimes'] += 1
    for m in [.5, 1, 2]:
        for k in [.5, 1, 3]:
            omega = math.sqrt(k/m)
            for gamma in [.05, .2, 1, 2, 5]:
                c = gamma*m*omega
                def integrand(z):
                    sn, co = math.sin(z), math.cos(z)
                    return c*omega*sn*sn/(m*m*omega*omega*math.cos(2*z)**2+c*c*sn*sn*co*co)
                total = adaptive(integrand,0,math.pi/4)+adaptive(integrand,math.pi/4,math.pi/2)
                assert abs(total-math.pi/(2*m)) < 2e-9
                counts['spectralIntegrals'] += 1
                for cutoff in [math.sqrt(2)*omega, 2*omega, 8*omega]:
                    tail = adaptive(integrand, math.atan(cutoff/omega), math.pi/2)
                    assert 0 < tail <= 4*c/(m*m*cutoff)+1e-10
                    counts['tailBounds'] += 1
                for ratio in [0, .3, 1, 2, 8]:
                    frequency = ratio*omega
                    weight = c*frequency**2/((k-m*frequency**2)**2+c*c*frequency**2)
                    chi = 1/complex(k-m*frequency**2,c*frequency)
                    assert weight >= 0 and abs(weight+frequency*chi.imag) < 1e-11
                    counts['positivePowerStates'] += 1
    for c in [.05,.2,1,4]:
        f = lambda w: 2/math.pi*c*w*w/((1-w*w)**2+c*c*w*w)
        for a,b in [(0,8),(.8,1.2)]:
            coarse, fine = simpson(f,a,b,10000), simpson(f,a,b,20000)
            assert abs(coarse-fine) < 1e-8 and 0 < fine < 1
            counts['finiteWindowIntegrals'] += 1
    return counts


if __name__ == '__main__':
    result = verify()
    print(json.dumps(result, sort_keys=True) if '--json' in sys.argv else 'Resonance research checks passed: ' + json.dumps(result, sort_keys=True))
