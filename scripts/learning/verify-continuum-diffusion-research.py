"""Independent finite-model entropy/information checks; standard library only.

Quadrature and finite differences are numerical diagnostics, not a certificate
for all parameters or arbitrary tails. Exact checks use rational moments.
"""
from fractions import Fraction as F
from itertools import product
from math import sin, cos, pi, sqrt, log, log1p, exp, isclose
import json


def check():
    counts = {}
    # Integrate the actual positive temperature and its gradient over a period.
    # C,V,Tmean,kappa,k are specified teaching scales in their SI units.
    n = 8000
    count = 0
    for a, phase, scale in product([-.2, -.1, 0., .05, .2], [0., .71], [1., 2.]):
        C, V, mean, kap, k = 2*scale, 3/scale, 300*scale, .7*scale, 1.2
        alpha = kap/C
        entropy = sensible = norm = prod = rate = 0.
        for j in range(n):
            theta = 2*pi*(j+.5)/n+phase
            T = mean*(1+a*cos(theta))
            Tx = -mean*a*k*sin(theta)
            Tt = -alpha*k*k*mean*a*cos(theta)
            entropy += C*log(T/273.15)*V/n
            sensible += C*(T-273.15)*V/n
            norm += .5*(T-mean)**2*V/n
            prod += kap*Tx*Tx/(T*T)*V/n
            rate += C*Tt/T*V/n
        r = sqrt(1-a*a)
        deficit = -log1p(-a*a/(2*(1+r)))
        closed_s = C*V*(log(mean/273.15)-deficit)
        closed_prod = kap*V*k*k*a*a/(r*(1+r))
        assert isclose(entropy, closed_s, abs_tol=2e-10, rel_tol=1e-12)
        assert isclose(sensible, C*V*(mean-273.15), abs_tol=2e-8, rel_tol=1e-12)
        assert isclose(norm, V*mean*mean*a*a/4, abs_tol=1e-9, rel_tol=1e-12)
        assert isclose(prod, closed_prod, abs_tol=2e-12)
        assert isclose(rate, prod, abs_tol=2e-12)
        assert prod >= 0
        assert (prod == 0) == (a == 0)
        count += 1
    counts['independentPhysicalHeatPeriodIntegrals'] = count

    # Direct finite differences of local s(T) and q(T)/T, not a derivative of
    # the already integrated closed entropy formula.
    count = 0
    C, mean, kap, k = 2., 300., .7, 1.2
    alpha = kap/C
    for a0, x, t in product([-.2, .1, .2], [.17, .61], [0., .4]):
        def fields(x, t):
            a = a0*exp(-alpha*k*k*t)
            T = mean*(1+a*cos(k*x))
            flux = kap*mean*a*k*sin(k*x)
            return C*log(T/273.15), flux/T
        eps = 1e-5
        st = (fields(x,t+eps)[0]-fields(x,t-eps)[0])/(2*eps)
        fx = (fields(x+eps,t)[1]-fields(x-eps,t)[1])/(2*eps)
        a = a0*exp(-alpha*k*k*t)
        production = kap*(a*k*sin(k*x)/(1+a*cos(k*x)))**2
        assert isclose(st+fx, production, abs_tol=1e-10)
        count += 1
    counts['independentLocalEntropyBalanceDifferences'] = count

    # Exact Gaussian score expectation with E[y]=0, E[y^2]=vp. Separately
    # differentiate KL algebraically using vp_dot=vq_dot=2D, delta_dot=0.
    count = 0
    for vp, vq, delta, D in product([F(1,2), F(1), F(2)],
                                     [F(2,5), F(1), F(3)],
                                     [F(-1), F(0), F(3,2)],
                                     [F(0), F(1,2), F(2)]):
        score_y, score_0 = 1/vq-1/vp, delta/vq
        fisher_moments = score_y**2*vp+score_0**2
        fisher_closed = (vp-vq)**2/(vp*vq*vq)+delta**2/(vq*vq)
        assert fisher_moments == fisher_closed >= 0
        derivative = D*(2/vq-vp/(vq*vq)-1/vp-delta**2/(vq*vq))
        assert derivative == -D*fisher_moments <= 0
        count += 1
    counts['exactGaussianScoreAndDissipationCases'] = count

    # Independent p-weighted log and score integrals in standardized p
    # coordinates, for moderate cases only; explicit finite tail truncation.
    count = 0
    for vp, vq, delta in product([.5,1.,2.], [.4,1.,3.], [-1.,0.,1.5]):
        numeric = fisher = 0.
        n = 16000
        for j in range(n):
            z = -11+22*(j+.5)/n
            y = sqrt(vp)*z
            weight = exp(-z*z/2)/sqrt(2*pi)*22/n
            log_ratio = .5*log(vq/vp)+(y+delta)**2/(2*vq)-z*z/2
            score = -y/vp+(y+delta)/vq
            numeric += weight*log_ratio
            fisher += weight*score*score
        closed = .5*(vp/vq-1-log(vp/vq)+delta*delta/vq)
        closed_f = (vp-vq)**2/(vp*vq*vq)+delta*delta/(vq*vq)
        assert isclose(numeric, closed, abs_tol=2e-12)
        assert isclose(fisher, closed_f, abs_tol=2e-12)
        count += 1
    counts['independentGaussianLogAndScoreIntegrals'] = count

    # Every correlation is attainable by a joint Gaussian; rho=1 uses the
    # shared-standard-normal coupling and attains the exact lower bound.
    count = 0
    for sp, sq, delta, corr in product([F(1,2), F(1), F(2)],
                                       [F(1,2), F(1), F(2)],
                                       [F(-1), F(0), F(3,2)],
                                       [F(-1), F(-1,2), F(0), F(1)]):
        cost = delta*delta+sp*sp+sq*sq-2*corr*sp*sq
        minimum = delta*delta+(sp-sq)**2
        assert cost >= minimum
        assert (cost == minimum) == (corr == 1)
        count += 1
    counts['exactQuadraticCouplingCostCases'] = count

    count = 0
    for delta, t in product([F(0), F(1), F(2)], [F(0), F(1), F(4), F(9), F(99)]):
        v = 1+t
        kl = delta*delta/(2*v)
        w2 = delta*delta
        assert 2*v*kl == w2
        if delta:
            assert w2 > 0
            assert kl <= delta*delta/2
        count += 1
    counts['exactPersistentCenterErrorCases'] = count

    # Different positive masses normalize to the same exact shape value.
    count = 0
    area, shape = F(3,2), F(2,7)
    for mp, mq in product([F(1), F(2), F(3,2)], repeat=2):
        cp, cq = mp*shape/area, mq*shape/area
        assert area*cp/mp == area*cq/mq == shape
        assert cp/cq == mp/mq
        assert (cp == cq) == (mp == mq)
        count += 1
    counts['exactNormalizedShapeMassBlindnessCases'] = count
    return counts


if __name__ == '__main__':
    print(json.dumps(check(), sort_keys=True))
