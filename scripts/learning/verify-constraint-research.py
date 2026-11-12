"""Stdlib checks for nonholonomic reactions and time-dependent Dirac motion."""
from fractions import Fraction as F
from itertools import product
from math import asinh, sqrt, sin, cos, isclose
import json
import sys

J = [[F(0), F(0), F(1), F(0)], [F(0), F(0), F(0), F(1)],
     [-F(1), F(0), F(0), F(0)], [F(0), -F(1), F(0), F(0)]]


def transpose(a):
    return list(map(list, zip(*a)))


def mul(a, b):
    return [[sum(x*y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def moving_circle(m, r0, c, angular, theta0, t):
    r = r0+c*t
    if m <= 0 or r0 <= 0 or r <= 0:
        raise ValueError('Positive mass and radius required throughout the interval.')
    theta = theta0+angular*t/(m*r0*r0) if c == 0 else theta0+angular*(1/r0-1/r)/(m*c)
    er = [cos(theta), sin(theta)]
    et = [-er[1], er[0]]
    return [r*x for x in er], [m*c*x+angular*y/r for x, y in zip(er, et)]


def main():
    reaction_states = vakonomic = loops = particle_paths = circles = circle_paths = rejected = 0
    for m, ell, x, u, vy in product([F(1,2), F(1), F(2)], [F(1,2), F(1), F(2)],
                                    [-2, -1, 0, 1, 2], [-1, 0, 1], [-1, 0, 1]):
        v = [u, F(vy), x*F(vy)/ell]
        nu = m*ell*u*vy/(ell*ell+x*x)
        a = [F(0), -x*u*vy/(ell*ell+x*x), ell*u*vy/(ell*ell+x*x)]
        assert [m*b for b in a] == [0, -nu*x/ell, nu]
        assert a[2]-u*v[1]/ell-x*a[1]/ell == 0
        assert m*dot(v,a) == 0
        for mu in [-F(1), F(0), F(1)]:
            mudot = -nu
            residual = [m*a[0]+mu*v[1]/ell,
                        m*a[1]-(x*mudot+mu*u)/ell, m*a[2]+mudot]
            assert residual == [mu*v[1]/ell, -mu*u/ell, 0]
            vakonomic += 1
        reaction_states += 1
    # A rectangular admissible xy loop has height area/ell; it need not solve dynamics.
    for ell, x0, width, height, sign in product([F(1,2), F(1), F(2)], [-1, 1],
                                               [F(1,2), F(2)], [F(1,2), F(2)], [-1, 1]):
        height_gain = ((x0+width)*sign*height-x0*sign*height)/ell
        assert height_gain == sign*width*height/ell
        loops += 1
    # Verify the analytic particle family by independent finite differences.
    for m, ell, u, v0, t in product([.5, 1., 2.], [.5, 1., 2.], [-1., 0., 1.],
                                    [-1., 0., 1.], [-1., 0., .5, 1., 2.]):
        def point(clock):
            d = sqrt(ell*ell+u*u*clock*clock)
            y = v0*clock if u == 0 else v0*ell*asinh(u*clock/ell)/u
            z = 0. if u == 0 else v0*(d-ell)/u
            return [u*clock, y, z], [u, v0*ell/d, v0*u*clock/d]
        q,v = point(t)
        h = 1e-5
        qa,va = point(t+h)
        qb,vb = point(t-h)
        numeric_v = [(a-b)/(2*h) for a,b in zip(qa,qb)]
        numeric_a = [(a-b)/(2*h) for a,b in zip(va,vb)]
        a = [0., -q[0]*u*v[1]/(ell*ell+q[0]**2), ell*u*v[1]/(ell*ell+q[0]**2)]
        assert all(isclose(a,b,rel_tol=1e-7,abs_tol=1e-8) for a,b in zip(v,numeric_v))
        assert all(isclose(a,b,rel_tol=1e-7,abs_tol=1e-8) for a,b in zip(a,numeric_a))
        assert abs(v[2]-q[0]*v[1]/ell) < 1e-12
        assert abs(m*dot(v,v)/2-m*(u*u+v0*v0)/2) < 1e-12
        particle_paths += 1
    for m,r,rd,rdd,angular,er in product([F(1,2),F(1),F(2)], [F(1,2),F(1),F(2)],
                                         [-F(1),F(0),F(1)], [-F(1),F(0),F(1)],
                                         [-F(1),F(0),F(1)],
                                         [(F(1),F(0)),(F(0),F(1)),(F(3,5),F(4,5)),(-F(5,13),F(12,13))]):
        et = [-er[1],er[0]]
        q = [r*x for x in er]
        p = [m*rd*x+angular*y/r for x,y in zip(er,et)]
        s = dot(q,q)
        g = [q+[F(0),F(0)], p+q]
        ci = [[F(0),-1/s],[1/s,F(0)]]
        correction = mul(mul(mul(mul(J,transpose(g)),ci),g),J)
        d = [[J[i][j]-correction[i][j] for j in range(4)] for i in range(4)]
        phi_t = [-r*rd,-m*(rd*rd+r*rdd)]
        affine = mul(mul(mul(J,transpose(g)),ci),[[x] for x in phi_t])
        grad = [F(0),F(0),p[0]/m,p[1]/m]
        static = [dot(row,grad) for row in d]
        flow = [static[i]-affine[i][0] for i in range(4)]
        assert flow[:2] == [x/m for x in p]
        reaction = [(m*rdd/r-angular*angular/(m*r**4))*x for x in q]
        assert flow[2:] == reaction
        assert [dot(row,flow)+partial for row,partial in zip(g,phi_t)] == [0,0]
        assert q[0]*flow[3]-q[1]*flow[2]+flow[0]*p[1]-flow[1]*p[0] == 0
        assert dot(reaction,flow[:2]) == m*rd*rdd-angular*angular*rd/(m*r**3)
        assert [dot(row,static)+partial for row,partial in zip(g,phi_t)] == phi_t
        circles += 1
    for m,r0,c,angular,theta0,t in product([1.,2.], [.5,1.,2.], [-.5,0.,.5],
                                          [-1.,0.,1.], [0.,.3], [0.,.25,.5,1.]):
        if r0+c*t <= 0:
            try:
                moving_circle(m,r0,c,angular,theta0,t)
                raise AssertionError('Collapsed circle must reject.')
            except ValueError:
                rejected += 1
            continue
        q,p = moving_circle(m,r0,c,angular,theta0,t)
        r = r0+c*t
        assert abs(dot(q,p)-m*r*c) < 1e-12
        assert abs(q[0]*p[1]-q[1]*p[0]-angular) < 1e-12
        assert abs(dot(p,p)/(2*m)-(m*c*c/2+angular*angular/(2*m*r*r))) < 1e-12
        h = 1e-5
        qa,pa = moving_circle(m,r0,c,angular,theta0,t+h)
        qb,pb = moving_circle(m,r0,c,angular,theta0,t-h)
        assert all(isclose((a-b)/(2*h),v/m,rel_tol=2e-6,abs_tol=2e-7) for a,b,v in zip(qa,qb,p))
        assert all(isclose((a-b)/(2*h),-angular*angular*v/(m*r**4),rel_tol=2e-6,abs_tol=2e-7) for a,b,v in zip(pa,pb,q))
        circle_paths += 1
    return dict(nonholonomicStates=reaction_states, vakonomicSnapshotChecks=vakonomic,
                admissibleLoopChecks=loops, nonholonomicPathStates=particle_paths,
                movingCircleExactStates=circles, movingCirclePathStates=circle_paths,
                collapsedCircleRejections=rejected)


if __name__ == '__main__':
    result = main()
    print(json.dumps(result,sort_keys=True) if '--json' in sys.argv else
          'Constraint research checks passed: '+json.dumps(result,sort_keys=True))
