"""Independent four-dimensional tensor, pushforward and density checks (stdlib)."""
from itertools import combinations, product
from math import exp, isclose
import json
import sys

J = [[0, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 0, 0], [0, -1, 0, 0]]


def tensor(z, length, uniform):
    f = exp(z[0]/length)
    return [[J[i][j]*(f if uniform or {i, j} == {0, 2} else 1)
             for j in range(4)] for i in range(4)]


def derivative(z, length, uniform, axis):
    f = exp(z[0]/length)
    return [[J[i][j]*f/length if axis == 0 and (uniform or {i, j} == {0, 2})
             else 0 for j in range(4)] for i in range(4)]


def jacobi(z, length, uniform, triple):
    pi = tensor(z, length, uniform)
    i, j, k = triple
    return sum(pi[i][a]*derivative(z, length, uniform, a)[j][k]
               + pi[j][a]*derivative(z, length, uniform, a)[k][i]
               + pi[k][a]*derivative(z, length, uniform, a)[i][j]
               for a in range(4))


def main():
    states = pushforwards = energy = density = 0
    for length, q1, q2, p1, p2 in product([.5, 1, 2], [-1, -.5, 0, .5, 1],
                                       [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]):
        f = exp(q1/length)
        z = [q1, q2, f*p1, p2]
        good, bad = tensor(z, length, False), tensor(z, length, True)
        df = [[1, 0, 0, 0], [0, 1, 0, 0],
              [f*p1/length, 0, f, 0], [0, 0, 0, 1]]
        pushed = [[sum(df[i][a]*J[a][b]*df[j][b] for a in range(4) for b in range(4))
                   for j in range(4)] for i in range(4)]
        assert all(isclose(pushed[i][j], good[i][j], abs_tol=1e-12)
                   for i in range(4) for j in range(4))
        pushforwards += 1
        for triple in combinations(range(4), 3):
            assert abs(jacobi(z, length, False, triple)) < 1e-12
        assert isclose(jacobi(z, length, True, (1, 3, 2)), -f*f/length)
        # Check the supplied analytic coefficient derivatives independently.
        h = 1e-5
        for uniform in [False, True]:
            for axis in range(4):
                za, zb = z[:], z[:]
                za[axis] += h
                zb[axis] -= h
                ta, tb = tensor(za, length, uniform), tensor(zb, length, uniform)
                d = derivative(z, length, uniform, axis)
                assert all(isclose((ta[i][j]-tb[i][j])/(2*h), d[i][j],
                                   rel_tol=1e-8, abs_tol=1e-9)
                           for i in range(4) for j in range(4))
        # Independently differentiate the nondegenerate form coefficients.
        # Convention: Omega=-Pi^{-1}; each 2x2 block has reciprocal factor.
        def form(point, uniform):
            factor = exp(point[0]/length)
            return [[J[i][j]/(factor if uniform or {i, j} == {0, 2} else 1)
                     for j in range(4)] for i in range(4)]
        for uniform in [False, True]:
            for i, j, k in combinations(range(4), 3):
                exterior = 0
                for axis, b, c in [(i, j, k), (j, k, i), (k, i, j)]:
                    za, zb = z[:], z[:]
                    za[axis] += h
                    zb[axis] -= h
                    exterior += (form(za, uniform)[b][c]-form(zb, uniform)[b][c])/(2*h)
                target = -1/(length*f) if uniform and (i, j, k) == (0, 1, 3) else 0
                assert isclose(exterior, target, rel_tol=1e-8, abs_tol=1e-9)
        # For H=P1 times a unit velocity, bad X=(f,0,0,0).
        # Its form-top density f^-2 is not invariant; density f^-1 is.
        top_div = (exp(-(q1+h)/length)-exp(-(q1-h)/length))/(2*h)
        assert isclose(top_div, -1/(length*f), rel_tol=1e-8, abs_tol=1e-9)
        # A coupled quadratic H, with fixed SI coefficients, tests both pairs.
        grad = [z[0]+z[1], z[0]+2*z[1], z[2]+z[3], z[2]+2*z[3]]
        for pi in [good, bad]:
            rate = [sum(row[j]*grad[j] for j in range(4)) for row in pi]
            assert abs(sum(a*b for a, b in zip(grad, rate))) < 1e-10
            energy += 1
        # rho=1/f gives div(rho*Pi grad H)=0 for BOTH tensors.
        # For bad: rho*Pi=J. Good: diag pair factors (1,1/f).
        # Verify with a numerical divergence, not just matrix skewness.
        for uniform in [False, True]:
            def field(point):
                x, y, p, r = point
                g = [x+y, x+2*y, p+r, p+2*r]
                mat = tensor(point, length, uniform)
                return [sum(row[j]*g[j] for j in range(4))/exp(x/length) for row in mat]
            div = 0
            for axis in range(4):
                za, zb = z[:], z[:]
                za[axis] += h
                zb[axis] -= h
                div += (field(za)[axis]-field(zb)[axis])/(2*h)
            assert abs(div) < 1e-7
            density += 1
        states += 1
    # H1 depends on the first pair only: it hides the wrong second block.
    probes = 0
    for q1 in [-1, 0, 1]:
        z = [q1, .2, 1.5, 1]
        good, bad = tensor(z, 1, False), tensor(z, 1, True)
        grad1 = [-exp(-2*q1)*z[2]**2, 0, exp(-2*q1)*z[2], 0]
        assert all(isclose(sum(good[i][j]*grad1[j] for j in range(4)),
                           sum(bad[i][j]*grad1[j] for j in range(4)), abs_tol=1e-12)
                   for i in range(4))
        assert good[1][3] == 1 and bad[1][3] == exp(q1)
        probes += 1
    return dict(tensorStates=states, pushforwardStates=pushforwards,
                energyChecks=energy, weightedDivergenceChecks=density,
                hiddenPairProbes=probes)


if __name__ == '__main__':
    result = main()
    print(json.dumps(result, sort_keys=True) if '--json' in sys.argv else
          'Four-dimensional research checks passed: '+json.dumps(result, sort_keys=True))
