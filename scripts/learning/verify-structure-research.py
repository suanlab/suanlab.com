"""Stdlib verification of exact quadratic invariants and state-selected clocks."""
from fractions import Fraction as F
from itertools import product
from math import asin, sin, cos, sqrt, pi, tanh, comb, isclose
from decimal import Decimal, localcontext
import json
import sys


def matrix(m,w,h):
    x = h*w
    return [[1-x*x/2,h/m],[-h*m*w*w*(1-x*x/4),1-x*x/2]]


def mul(a,b):
    return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]


def apply(a,z):
    return [sum(x*y for x,y in zip(row,z)) for row in a]


def invariant(q,p,m,w,h):
    return p*p/(2*m)+m*w*w*(1-h*h*w*w/4)*q*q/2


def main():
    matrices = rational = closed = boundary = generators = clocks = rattle = trajectories = 0
    for m,w,x in product([F(1,2),F(1),F(2)],[F(1,2),F(1),F(2)],
                           [F(1,4),F(1,2),F(1),F(3,2),F(2),F(5,2)]):
        h = x/w
        a = matrix(m,w,h)
        assert a[0][0]*a[1][1]-a[0][1]*a[1][0] == 1
        k = [[m*w*w*(1-x*x/4),F(0)],[F(0),1/m]]
        assert mul(mul(list(map(list,zip(*a))),k),a) == k
        matrices += 1
        for q0,p0,n in product([-F(1),F(0),F(1)],[-F(1),F(0),F(1)],[0,1,2,8]):
            z = [q0,p0]
            for _ in range(n):
                half = z[1]-h*m*w*w*z[0]/2
                q = z[0]+h*half/m
                p = half-h*m*w*w*q/2
                assert [q,p] == apply(a,z)
                z = [q,p]
            assert invariant(*z,m,w,h) == invariant(q0,p0,m,w,h)
            rational += 1
        if x < 2:
            mf,wf,hf = float(m),float(w),float(h)
            theta = 2*asin(float(x)/2)
            kap = 1-float(x*x)/4
            scale = theta/sin(theta)
            frequency = scale*wf*sqrt(kap)
            flow = [[cos(hf*frequency),sin(hf*frequency)*scale/(mf*frequency)],
                    [-sin(hf*frequency)*scale*mf*wf*wf*kap/frequency,cos(hf*frequency)]]
            assert all(isclose(flow[i][j],float(a[i][j]),abs_tol=1e-12) for i in range(2) for j in range(2))
            generators += 1
            for q0,p0,n in product([-1.,0.,1.],[-1.,0.,1.],[0,1,2,8,32]):
                z = [q0,p0]
                for _ in range(n):
                    z = apply([[float(v) for v in row] for row in a],z)
                target = [q0*cos(n*theta)+p0*sin(n*theta)/(mf*wf*sqrt(kap)),
                          p0*cos(n*theta)-mf*wf*sqrt(kap)*q0*sin(n*theta)]
                assert all(isclose(u,v,abs_tol=2e-11) for u,v in zip(z,target))
                closed += 1
        if x == 2:
            for q0,p0,n in product([-F(1),F(0),F(1)],[-F(1),F(0),F(1)],[0,1,2,8,32]):
                z = [q0,p0]
                for _ in range(n):
                    z = apply(a,z)
                assert z == [(-1)**n*(q0-n*h*p0/m),(-1)**n*p0]
                boundary += 1
    long_horizons = 0
    for h in [.2,.1,.05,.025]:
        n = round(24/h**3)
        error = n*(2*asin(h/2)-h)
        with localcontext() as context:
            context.prec = 70
            hd = Decimal(str(h))
            theta = 2*sum(Decimal(comb(2*j,j))*(hd/2)**(2*j+1)/(Decimal(4)**j*(2*j+1)) for j in range(50))
            reference = Decimal(n)*(theta-hd)
        assert abs(error-float(reference)) < 2e-10
        assert 1 < error < 1.01
        assert n*h == 24/(h*h) or isclose(n*h,24/(h*h))
        long_horizons += 1
    for m,r,L,h0,eps,theta in product([.5,1.,2.],[.5,1.,2.],[-1.,0.,1.],
                                     [.25,.5,3.],[0.,.25,.5],[-pi,-pi/2,0.,pi/2,pi,3*pi/2]):
        inertia = m*r*r
        def step(angle,momentum):
            elapsed = h0*(1+eps*sin(angle))
            return angle+momentum*elapsed/inertia,momentum,elapsed
        final,momentum,elapsed = step(theta,L)
        assert elapsed > 0
        q = [r*cos(final),r*sin(final)]
        p = [-L*sin(final)/r,L*cos(final)/r]
        assert isclose(q[0]*p[1]-q[1]*p[0],L,abs_tol=1e-12)
        assert abs(sum(a*b for a,b in zip(q,p))) < 1e-12
        assert isclose(sum(a*a for a in p)/(2*m),L*L/(2*inertia),abs_tol=1e-12)
        d = 1e-6
        numeric = (step(theta+d,L)[0]-step(theta-d,L)[0])/(2*d)
        determinant = 1+L*h0*eps*cos(theta)/inertia
        assert isclose(numeric,determinant,rel_tol=1e-7,abs_tol=2e-8)
        # A momentum-only clock keeps the theta derivative equal to one.
        def momentum_clock(angle):
            return angle+L*h0*(1+eps*tanh(L))/inertia
        assert isclose((momentum_clock(theta+d)-momentum_clock(theta-d))/(2*d),1,abs_tol=2e-8)
        clocks += 1
    for L,h0,eps,theta in product([-1.,0.,1.],[.1,.5],[0.,.25,.5],[-pi,-pi/2,0.,pi/2,pi,3*pi/2]):
        def update(angle):
            return angle+asin(L*h0*(1+eps*sin(angle)))
        nu = L*h0*(1+eps*sin(theta))
        assert abs(nu) <= .75
        expected = 1+L*h0*eps*cos(theta)/sqrt(1-nu*nu)
        d = 1e-6
        numeric = (update(theta+d)-update(theta-d))/(2*d)
        assert isclose(numeric,expected,rel_tol=1e-8,abs_tol=1e-8)
        rattle += 1
    lo,hi = .1,pi
    for _ in range(80):
        s = (lo+hi)/2
        if s-1.5*sin(s) < 0:
            lo = s
        else:
            hi = s
    s = (lo+hi)/2
    times = []
    for theta in [pi-s,pi,pi+s]:
        elapsed = 3*(1+.5*sin(theta))
        assert abs(theta+elapsed-(pi+3)) < 1e-12
        assert elapsed > 0
        assert abs((pi+3)-elapsed-theta) < 1e-12
        times.append(elapsed)
    assert len(set(times)) == 3
    assert isclose(1+1.5*cos(pi),-.5)
    for initial,n in product([0.,.5,pi],[1,2,8,32]):
        theta,total = initial,0.
        for _ in range(n):
            elapsed = 3*(1+.5*sin(theta))
            total += elapsed
            theta += elapsed
        assert abs(theta-initial-total) < 1e-11
        trajectories += 1
    return dict(harmonicMatrixCases=matrices, exactInvariantStates=rational,
                closedFormStates=closed, modifiedGeneratorCases=generators,
                stabilityBoundaryStates=boundary, longHorizonCases=long_horizons,
                observationClockStates=clocks, controlledRattleStates=rattle,
                foldCollisionStates=3, accumulatedClockTrajectories=trajectories)


if __name__ == '__main__':
    result = main()
    print(json.dumps(result,sort_keys=True) if '--json' in sys.argv else
          'Structure research checks passed: '+json.dumps(result,sort_keys=True))
