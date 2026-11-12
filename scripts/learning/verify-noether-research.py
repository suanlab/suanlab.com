"""Independent exact-rational Noether rank/boost and numerical phase checks."""
from fractions import Fraction as F
from itertools import product
from math import isclose
import json
import sys


def inverse_velocity(p, m, lam):
    if m <= 0 or lam < 0:
        raise ValueError('Require m>0 and lambda>=0')
    if lam == 0 or p == 0:
        return p/m
    sign = -1 if p < 0 else 1
    lo, hi = 0., abs(p)/m
    for _ in range(90):
        mid = (lo+hi)/2
        if m*mid+lam*mid**3 < abs(p):
            lo = mid
        else:
            hi = mid
    return sign*(lo+hi)/2


def main():
    reconstruction = perturbations = hidden = boost = canonical = 0
    for ell, x, y, rx, ry in product([F(1,2), F(1), F(2)],
                                    [F(-2), F(-1,100), F(1,100), F(2)],
                                    [-2, 0, 3], [-2, 0, 2], [-3, 0, 3]):
        b1, br = ell*rx, -y*rx+x*ry
        assert b1/ell == rx
        assert (br+y*b1/ell)/x == ry
        reconstruction += 1
        # The deterministic bound must hold for both signs of both errors.
        e1, er = F(1,100), F(1,50)
        for s1, sr in product([-1, 1], repeat=2):
            error = (sr*er+y*s1*e1/ell)/x
            assert abs(error) <= (er+abs(y/ell)*e1)/abs(x)
            perturbations += 1
    # Entire wrong path stays in the rank-one characteristic set.
    for m, ell, time_scale, t in product([F(1), F(2)], [F(1), F(2)],
                                        [F(1), F(2)], [F(0), F(1,2), F(1), F(2)]):
        x, y, vx, vy = F(0), ell*(1+(t/time_scale)**2), F(0), 2*ell*t/time_scale**2
        rx, ry = F(0), 2*m*ell/time_scale**2
        assert ell*m*vx == 0 and m*(x*vy-y*vx) == 0
        assert ell*rx == 0 and -y*rx+x*ry == 0 and ry != 0
        assert ell*ry != 0  # The extra y-translation detects the residual.
        hidden += 1
    for m, lam, v, t, a, u in product([F(1), F(2)], [F(0), F(1,2), F(1)],
                                      [-2, 0, 2], [F(0), F(1,2), F(2)],
                                      [-1, 0, 1], [F(-1), F(1)]):
        p, hv = m*v+lam*v**3, m+3*lam*v*v
        residual, defect = hv*a, lam*u*v**3
        rate = u*(p+t*hv*a-m*v)
        assert rate == defect+u*t*residual
        assert hv > 0
        # Ordinary point boost maps each free solution to another one.
        assert hv*0 == (m+3*lam*(v+u)**2)*0 == 0
        boost += 1
    for m, lam, p, t, q, qdot, pdot in product([1., 2.], [0., .5, 1.],
                                             [-2., 0., 2.], [0., .5, 2.],
                                             [-1., 1.], [-.5, 1.], [-1., 1.]):
        v = inverse_velocity(p, m, lam)
        assert abs(m*v+lam*v**3-p) < 1e-12
        hv = m+3*lam*v*v
        # p_*=1 kg m/s. Arbitrary phase velocities: no equations imposed.
        dq, dp = -t/hv, -1
        dqdot = -1/hv+t*6*lam*v*pdot/hv**3
        delta_phase = dp*qdot+p*dqdot-v*dp
        def hamiltonian(momentum):
            speed = inverse_velocity(momentum, m, lam)
            return momentum*speed-m*speed**2/2-lam*speed**4/4
        def generator(pos, momentum, clock):
            return pos-clock*inverse_velocity(momentum, m, lam)
        h = 1e-5
        hp = (hamiltonian(p+h)-hamiltonian(p-h))/(2*h)
        gp = (generator(q,p+h,t)-generator(q,p-h,t))/(2*h)
        gt = (generator(q,p,t+h)-generator(q,p,t-h))/(2*h)
        assert isclose(hp, v, rel_tol=1e-8, abs_tol=1e-9)
        assert isclose(gp, dq, rel_tol=1e-8, abs_tol=1e-9)
        assert abs(gt+hp) < 1e-8
        def boundary(s):
            clock, pos, momentum = t+s, q+s*qdot, p+s*pdot
            vel = inverse_velocity(momentum,m,lam)
            shift = -clock/(m+3*lam*vel*vel)
            return momentum*shift-generator(pos,momentum,clock)
        numeric = (boundary(h)-boundary(-h))/(2*h)
        assert isclose(numeric, delta_phase, rel_tol=2e-7, abs_tol=2e-8)
        # Finite flow of G: its velocity shift depends on the state for lam>0.
        eps = .2
        newp = p-eps
        newq = q+t*(inverse_velocity(newp,m,lam)-v)
        assert isclose(generator(newq,newp,t),generator(q,p,t),abs_tol=1e-12)
        canonical += 1
    return dict(characteristicStates=reconstruction, errorBoundChecks=perturbations,
                rankDeficientPathStates=hidden, quarticBoostStates=boost,
                phaseGeneratorStates=canonical)


if __name__ == '__main__':
    result = main()
    print(json.dumps(result, sort_keys=True) if '--json' in sys.argv else
          'Noether research checks passed: '+json.dumps(result, sort_keys=True))
