"""Independent checks of the declared reaction and comparison models.

Standard library only. Numerical finite cases do not prove the comparison
principle, nor simulate a nonlinear KPP solution or certify arbitrary tails.
"""
from fractions import Fraction as F
from itertools import product
from math import exp, log, pi, sqrt, cos, sin, isclose
import json


def matrix(kf, kb, q):
    pa, pb = kb/(kf+kb), kf/(kf+kb)
    return [[pa+pb*q, pa*(1-q)], [pb*(1-q), pb+pa*q]]


def multiply(a, b):
    return [[sum(a[i][l]*b[l][j] for l in [0, 1])
             for j in [0, 1]] for i in [0, 1]]


def mixture(x, t, kf=1., kb=1., U=0., D=.2):
    # Initial physical masses 1 kg each, area 1 m^2, sigma=.5 m.
    v = .25+2*D*t
    lam = kf+kb
    q = exp(-lam*t)
    e = matrix(kf, kb, q)
    g, gx, gt = [], [], []
    for mu in [-1+U*t, 1+U*t]:
        y = x-mu
        val = exp(-y*y/(2*v))/sqrt(2*pi*v)
        g.append(val)
        gx.append(-y/v*val)
        gt.append((U*y/v+D*(y*y/(v*v)-1/v))*val)
    c = [sum(e[i][j]*g[j] for j in [0, 1]) for i in [0, 1]]
    cx = [sum(e[i][j]*gx[j] for j in [0, 1]) for i in [0, 1]]
    source = -kf*c[0]+kb*c[1]
    ct = [sum(e[i][j]*gt[j] for j in [0, 1])+([source, -source][i])
          for i in [0, 1]]
    return c, cx, ct


def check():
    counts = {}
    rates = [F(1, 2), F(1), F(2)]
    qs = [F(0), F(1, 4), F(1, 2), F(1)]
    count = 0
    for kf, kb, q1, q2 in product(rates, rates, qs, qs):
        a, b = matrix(kf, kb, q1), matrix(kf, kb, q2)
        assert multiply(a, b) == matrix(kf, kb, q1*q2)
        assert all(val >= 0 for row in a for val in row)
        assert all(a[0][j]+a[1][j] == 1 for j in [0, 1])
        assert matrix(kf, kb, F(1)) == [[1, 0], [0, 1]]
        count += 1
    counts['exactReactionSemigroupCases'] = count

    count = 0
    for a, b, ax, bx in product([F(1, 5), F(1), F(3)],
                               [F(1, 5), F(1), F(3)],
                               [F(-2), F(0), F(3, 2)],
                               [F(-2), F(0), F(3, 2)]):
        total = a+b
        fraction = a/total
        fx = (b*ax-a*bx)/(total*total)
        hessian = ax*ax/a+bx*bx/b-(ax+bx)**2/total
        assert hessian == total*fx*fx/(fraction*(1-fraction)) >= 0
        count += 1
    counts['exactConditionalDiffusionHessianCases'] = count

    count = 0
    for kf, kb, a, b in product(rates, rates, [F(1,5),F(1),F(3)],
                                [F(1,5),F(1),F(3)]):
        pa, pb = kb/(kf+kb), kf/(kf+kb)
        assert (a/(pa*(a+b)))/(b/(pb*(a+b))) == kf*a/(kb*b)
        production = float(kf*a-kb*b)*log(float(kf*a/(kb*b)))
        assert production >= 0
        count += 1
    counts['exactReactionLogRatioAndSignCases'] = count

    # Compare actual mixture evaluations with independent spatial/time
    # differences. A one-sided derivative keeps t=0 inside the model domain.
    count = 0
    for kf, kb in [(1.,1.),(.4,1.3)]:
        for x, t in product([-1.,0.,1.], [0.,.2,.8]):
            U, D, eps = .3, .2, 1e-4
            c, _, _ = mixture(x,t,kf,kb,U,D)
            xp = mixture(x+eps,t,kf,kb,U,D)[0]
            xm = mixture(x-eps,t,kf,kb,U,D)[0]
            tp = mixture(x,t+eps,kf,kb,U,D)[0]
            tm = mixture(x,t-eps,kf,kb,U,D)[0] if t else None
            tt = mixture(x,2*eps,kf,kb,U,D)[0] if not t else None
            for j in [0,1]:
                dt = (tp[j]-tm[j])/(2*eps) if t else (-3*c[j]+4*tp[j]-tt[j])/(2*eps)
                dx = (xp[j]-xm[j])/(2*eps)
                dxx = (xp[j]-2*c[j]+xm[j])/(eps*eps)
                source = (-kf*c[0]+kb*c[1]) * (1 if j == 0 else -1)
                assert abs(dt+U*dx-D*dxx-source) < 3e-6
            count += 1
    counts['independentMixturePdeDifferenceCases'] = count

    count = 0
    for t in [0.,.1,.4,1.]:
        n, width = 12000, 14.
        hdot = diffusion = reaction = ma = mb = 0.
        for j in range(n):
            x = -7+width*(j+.5)/n
            (a,b), (ax,bx), (at,bt) = mixture(x,t)
            total = a+b
            hdot += (at*log(2*a/total)+bt*log(2*b/total))*width/n
            # Positive-component formula avoids subtracting f from one.
            diffusion += .2*(b*ax-a*bx)**2/(a*b*total)*width/n
            reaction += (a-b)*(log(a)-log(b))*width/n
            ma += a*width/n
            mb += b*width/n
        assert abs(hdot+diffusion+reaction) < 2e-8
        assert abs(ma-1) < 1e-12 and abs(mb-1) < 1e-12
        assert diffusion > 0 and reaction > 0
        count += 1
    counts['independentConditionalInformationBudgets'] = count

    # Scaled compact subsolutions, including signed relative speeds. Direct
    # derivatives and independent finite differences check the interior cost.
    count = 0
    for D, r, factor in product([.5,1.,2.], [.4,1.,1.5], [-1.8,-1.,0.,1.,1.8]):
        speed = factor*sqrt(D*r)
        gap = r-speed*speed/(4*D)
        L = pi*sqrt(D/(2*gap))
        eig = D*(pi/(2*L))**2+speed*speed/(4*D)
        sigma = sqrt(D/r)
        m0 = .2*exp(-L*L/(2*sigma*sigma))
        epsilon = .5*exp(-abs(speed)*L/(2*D))*min(m0,(r-eig)/r)
        assert epsilon > 0 and eig < r
        assert L > pi*D/sqrt(4*D*r-speed*speed)
        a, b = speed/(2*D), pi/(2*L)
        def shape(y):
            return exp(-a*y)*cos(b*y)
        for j in range(201):
            y = -L+2*L*(j+.5)/201
            w = shape(y)
            wp = exp(-a*y)*(-a*cos(b*y)-b*sin(b*y))
            wpp = exp(-a*y)*((a*a-b*b)*cos(b*y)+2*a*b*sin(b*y))
            value = epsilon*w
            assert 0 < value <= .2*exp(-y*y/(2*sigma*sigma))
            assert value <= (r-eig)/r
            assert isclose(-D*wpp-speed*wp,eig*w,abs_tol=2e-12,rel_tol=1e-11)
            assert -speed*wp-D*wpp-r*w+r*epsilon*w*w < 0
        for fraction in [-.8,-.4,0.,.4,.8]:
            y = L*fraction
            step = L*1e-4
            first = (shape(y+step)-shape(y-step))/(2*step)
            second = (shape(y+step)-2*shape(y)+shape(y-step))/(step*step)
            assert isclose(-D*second-speed*first,eig*shape(y),rel_tol=3e-6,abs_tol=3e-6)
        # Positive derivative jumps add negative diffusion point masses to
        # the weak logistic residual of the zero extension.
        assert epsilon*exp(a*L)*b > 0 and epsilon*exp(-a*L)*b > 0
        # An insufficient width fails already at the positive center.
        narrow = L/3
        narrow_cost = D*(pi/(2*narrow))**2+speed*speed/(4*D)
        assert narrow_cost > r
        count += 1
    counts['independentCompactMovingSubsolutionCases'] = count

    count = 0
    for D, r, factor in product([.5,1.,2.], [.4,1.,1.5], [-2.5,-1.,0.,1.,2.5]):
        relative = factor*sqrt(D*r)
        sigma2 = D/r
        initial_peak = .2
        expected = r-relative*relative/(4*D)
        t = 1e5/r
        variance = sigma2+2*D*t
        # Log of the actual linear Gaussian along its moving ray; c*=1 in
        # the declared SI concentration scale. No nonlinear field is solved.
        logval = log(initial_peak)+.5*log(sigma2/variance)+r*t-relative**2*t*t/(2*variance)
        assert abs(logval/t-expected) < 2e-4*r
        beta = sqrt(2*r/(3*D))
        assert isclose(D*beta+r/beta,5*sqrt(D*r/6),rel_tol=1e-14)
        assert 5*sqrt(D*r/6) > 2*sqrt(D*r)
        count += 1
    counts['independentLinearGaussianRayAndTailCases'] = count
    return counts


if __name__ == '__main__':
    print(json.dumps(check(), sort_keys=True))
