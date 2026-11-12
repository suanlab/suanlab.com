"""Independent residual-derivative and Hopf-Lax branch checks (stdlib only)."""
from fractions import Fraction as F
from itertools import product
from math import sin, cos, pi, sqrt, isclose
import json
import sys


def stationary_roots(x,t,a=1.,b=1.,m=1.):
    if min(a,b,m,t) <= 0:
        raise ValueError('Require a,b,m,t>0')
    linear,constant = m/t-b,-m*x/t
    def polynomial(y):
        return a*y**3+linear*y+constant
    bound = 1+max(abs(linear/a),abs(constant/a))
    cuts = [-bound,bound]
    if linear < 0:
        turn = sqrt(-linear/(3*a))
        cuts = [-bound,-turn,turn,bound]
    roots = [c for c in cuts if abs(polynomial(c)) < 1e-13*(1+abs(a*c**3)+abs(linear*c)+abs(constant))]
    for lo,hi in zip(cuts,cuts[1:]):
        if polynomial(lo)*polynomial(hi) >= 0:
            continue
        for _ in range(80):
            mid = (lo+hi)/2
            if polynomial(lo)*polynomial(mid) <= 0:
                hi = mid
            else:
                lo = mid
        roots.append((lo+hi)/2)
    unique = []
    for r in sorted(roots):
        if not unique or abs(r-unique[-1]) > 1e-9*max(1,abs(r)):
            unique.append(r)
    return unique


def initial(y,a,b):
    return a*y**4/4-b*y*y/2


def value(x,t,a=1.,b=1.,m=1.):
    roots = stationary_roots(x,t,a,b,m)
    return min(initial(y,a,b)+m*(x-y)**2/(2*t) for y in roots)


def main():
    identities = singular = oscillatory = sequence = minima = pde = shocks = 0
    # Generic quadratic S with mixed time dependence, in fixed SI reference units.
    for A,f,t,B,beta,alpha,m,k,force in product([F(1,100),F(1,2),F(1)],
             [-F(1,2),F(0),F(1,2)], [F(0),F(1,2),F(1)], [F(0),F(1)],
             [-F(1),F(1)], [-F(1),F(0),F(1)], [F(1),F(2)], [F(0),F(1)], [-F(1),F(1)]):
        mixed = A+f*t
        if mixed == 0:
            singular += 1
            continue
        C,E,D,g = F(-1,2),F(1,2),F(1,2),F(1)
        q = (beta-(C+E*t)*alpha)/mixed
        qdot = (-E*alpha-f*q)/mixed
        p = mixed*alpha+(B+D*t)*q+g*t*t/2
        pdot = f*alpha+D*q+(B+D*t)*qdot+g*t
        ra = f*q+E*alpha+(p/m)*mixed
        rq = D*q+f*alpha+g*t+(p/m)*(B+D*t)+k*q-force
        assert qdot-p/m == -ra/mixed
        assert pdot+k*q-force == rq-(B+D*t)*ra/mixed
        identities += 1
    for m,ell,T,scale,eps,alpha,q,tau in product([.5,1.,2.],repeat=8):
        # Convert the fifth grid coordinate to a small dimensionless epsilon.
        eps /= 10
        alpha -= 1
        q -= 1
        tau = (tau-.5)/1.5
        t = T*tau
        phase = q/(ell*eps)
        momentum = alpha+scale*eps*tau*cos(phase)
        st = -alpha*alpha/(2*m)+scale*ell*eps*eps*sin(phase)/T
        residual = st+momentum*momentum/(2*m)
        bound = scale*ell*eps*eps/T+abs(alpha)*scale*eps/m+scale*scale*eps*eps/(2*m)
        assert abs(residual) <= bound+1e-12
        ra = scale*eps*tau*cos(phase)/m
        rq = scale*eps*cos(phase)/T-momentum*scale*tau*sin(phase)/(m*ell)
        sqq = -scale*tau*sin(phase)/ell
        velocity_error = -scale*eps*tau*cos(phase)/m
        force_error = scale*eps*cos(phase)/T-scale*alpha*t*sin(phase)/(m*ell*T)
        assert abs(velocity_error+ra) < 1e-12
        assert abs(force_error-rq+sqq*ra) < 1e-12
        def s(position,clock):
            return alpha*position-alpha*alpha*clock/(2*m)+scale*ell*eps*eps*clock*sin(position/(ell*eps))/T
        h = min(T,ell*eps)*1e-5
        numeric_st = (s(q,t+h)-s(q,t-h))/(2*h)
        numeric_p = (s(q+h,t)-s(q-h,t))/(2*h)
        assert isclose(numeric_st,st,rel_tol=1e-7,abs_tol=1e-8)
        assert isclose(numeric_p,momentum,rel_tol=1e-7,abs_tol=1e-8)
        def path_p(dt):
            return alpha+scale*eps*(t+dt)*cos((q+alpha*dt/m)/(ell*eps))/T
        numeric_force = (path_p(h)-path_p(-h))/(2*h)
        assert isclose(numeric_force,force_error,rel_tol=1e-7,abs_tol=1e-8)
        oscillatory += 1
    for m,ell,T,scale,n in product([.5,1.,2.],[.5,1.,2.],[.5,1.,2.],[.5,1.,2.],[0,1,2,4,8,16,32]):
        eps = 1/(pi/2+2*pi*n)
        alpha = m*ell/T
        phase = 1/eps
        error = scale*eps*cos(phase)/T-scale*sin(phase)/T
        momentum = alpha+scale*eps*cos(phase)
        residual = -alpha*alpha/(2*m)+scale*ell*eps*eps*sin(phase)/T+momentum*momentum/(2*m)
        assert isclose(error,-scale/T,abs_tol=1e-12)
        assert isclose(momentum,alpha,abs_tol=1e-12)
        assert isclose(residual,scale*ell*eps*eps/T,abs_tol=1e-12)
        sequence += 1
    for a,b,m,tau,x in product([.5,1.,2.],[.5,1.,2.],[.5,1.,2.], [.5,.8,1.,1.25,2.,4.],[-1.,-.1,0.,.1,1.]):
        t = tau*m/b
        roots = stationary_roots(x,t,a,b,m)
        assert roots
        costs = [initial(y,a,b)+m*(x-y)**2/(2*t) for y in roots]
        best = min(costs)
        selected = [y for y,c in zip(roots,costs) if abs(c-best) < 1e-10]
        for y in roots:
            assert abs(a*y**3+(m/t-b)*y-m*x/t) < 1e-10
            p0 = a*y**3-b*y
            assert isclose(y+t*p0/m,x,abs_tol=1e-10)
        for y in selected:
            assert 3*a*y*y+m/t-b >= -1e-10
            assert initial(y,a,b) < max(0,a/4-b/2)+1
        if x == 0:
            expected = 0 if tau <= 1 else -(b-m/t)**2/(4*a)
            assert isclose(best,expected,abs_tol=1e-12)
            assert len(selected) == (2 if tau > 1 else 1)
        else:
            assert len(selected) == 1 and selected[0]*x > 0
        # Independent grid checks both ordinary and bounded-data localized objectives.
        bound = 1+max(abs((m/t-b)/a),abs(m*x/(a*t)))
        cap = max(0,a/4-b/2)+1
        for j in range(401):
            y = -bound+2*bound*j/400
            kinetic = m*(x-y)**2/(2*t)
            assert initial(y,a,b)+kinetic >= best-1e-10
            assert min(initial(y,a,b),cap)+kinetic >= best-1e-10
        if x:
            h = 1e-5
            vx = (value(x+h,t,a,b,m)-value(x-h,t,a,b,m))/(2*h)
            vt = (value(x,t+h,a,b,m)-value(x,t-h,a,b,m))/(2*h)
            assert abs(vt+vx*vx/(2*m)) < 2e-6
            pde += 1
        minima += 1
    for a,b,m,tau in product([F(1,2),F(1),F(2)],[F(1,2),F(1),F(2)],
                              [F(1,2),F(1),F(2)],[F(5,4),F(3,2),F(2),F(4)]):
        t = tau*m/b
        slope_squared = m*m*(b-m/t)/(a*t*t)
        vt = -m*(b-m/t)/(2*a*t*t)
        for gamma in [-F(1),-F(1,2),F(0),F(1,2),F(1)]:
            assert vt+gamma*gamma*slope_squared/(2*m) <= 0
        shocks += 1
    return dict(characteristicIdentityStates=identities, singularMixedHessianCases=singular,
                oscillatoryStates=oscillatory, smallResidualCounterexamples=sequence,
                hopfLaxStates=minima, differentiablePdeChecks=pde, shockTests=shocks)


if __name__ == '__main__':
    result = main()
    print(json.dumps(result,sort_keys=True) if '--json' in sys.argv else
          'Hamilton-Jacobi research checks passed: '+json.dumps(result,sort_keys=True))
