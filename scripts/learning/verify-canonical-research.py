"""Standard-library checks for Legendre inversion, nonlinear charts and reduction."""
from fractions import Fraction as Q
from itertools import product
from math import sqrt, exp, log, sin, cos
import json
import sys


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def speed(momentum, mass, quartic):
    if momentum < 0 or mass <= 0 or quartic < 0:
        raise ValueError('Require P>=0, m>0 and lambda>=0.')
    if momentum == 0:
        return 0.
    if quartic == 0:
        return momentum/mass
    lo, hi = 0., momentum/mass
    for _ in range(90):
        mid = (lo+hi)/2
        if mass*mid+quartic*mid**3 < momentum:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def hamiltonian(mass, quartic, radius, pr, angular, stiffness=0):
    if radius <= 0:
        raise ValueError('Polar chart requires positive radius.')
    magnitude = sqrt(pr*pr+angular*angular/(radius*radius))
    u = speed(magnitude, mass, quartic)
    return mass*u*u/2+3*quartic*u**4/4+stiffness*radius*radius/2


def legendre_checks():
    count = 0
    for mass, quartic, radius, velocity, stiffness in product(
            [.5, 1, 2], [0, .5, 1], [.5, 1, 2],
            [(0, 0), (1, 0), (0, 1), (1, 1), (-1, 2)], [0, 1]):
        vr, vt = velocity
        u2 = vr*vr+vt*vt
        pr, angular = (mass+quartic*u2)*vr, (mass+quartic*u2)*radius*vt
        magnitude = sqrt(pr*pr+angular*angular/radius**2)
        u = speed(magnitude, mass, quartic)
        assert abs(u-sqrt(u2)) < 2e-13
        h = hamiltonian(mass, quartic, radius, pr, angular, stiffness)
        lagrangian = mass*u2/2+quartic*u2*u2/4-stiffness*radius*radius/2
        assert abs(pr*vr+angular*vt/radius-lagrangian-h) < 3e-12
        step = 1e-5
        f = lambda r, p, ell: hamiltonian(mass, quartic, r, p, ell, stiffness)
        derivatives = [(f(radius, pr+step, angular)-f(radius, pr-step, angular))/(2*step),
                       (f(radius, pr, angular+step)-f(radius, pr, angular-step))/(2*step),
                       (f(radius+step, pr, angular)-f(radius-step, pr, angular))/(2*step)]
        expected_r = stiffness*radius-(u*angular*angular/(radius**3*magnitude) if magnitude else 0)
        targets = [vr, vt/radius, expected_r]
        assert max(abs(a-b) for a, b in zip(derivatives, targets)) < 3e-6
        if quartic:
            numeric = (hamiltonian(mass, quartic+step, radius, pr, angular, stiffness)
                       -hamiltonian(mass, quartic-step, radius, pr, angular, stiffness))/(2*step)
            assert abs(numeric+u**4/4) < 2e-6
        count += 1
    assert speed(0, 1, 1) == 0
    try:
        hamiltonian(1, 1, 0, 0, 0)
        raise AssertionError('Origin must remain outside the polar chart.')
    except ValueError:
        pass
    return count


def nonlinear_checks():
    count = 0
    for mass, stiffness, length, q, p in product([1, 2], [0, 1], [.5, 1, 2], [-1, 0, 1], [-1, 0, 1]):
        factor = exp(q/length)
        P = factor*p
        hp = exp(-2*q/length)*P/mass
        hq = -exp(-2*q/length)*P*P/(mass*length)+stiffness*q
        rate = [factor*hp, -factor*hq]
        transformed = [p/mass, factor*(-stiffness*q+p*p/(mass*length))]
        assert max(abs(a-b) for a, b in zip(rate, transformed)) < 1e-11
        assert abs(hq*rate[0]+hp*rate[1]) < 1e-11
        step = 1e-5
        def field(x, momentum):
            f = exp(x/length)
            return [momentum/(mass*f), momentum*momentum/(mass*length*f)-f*stiffness*x]
        div = ((field(q+step, P)[0]-field(q-step, P)[0])
               +(field(q, P+step)[1]-field(q, P-step)[1]))/(2*step)
        assert abs(div-rate[0]/length) < 2e-7
        def weighted(x, momentum):
            return [v/exp(x/length) for v in field(x, momentum)]
        weighted_div = ((weighted(q+step, P)[0]-weighted(q-step, P)[0])
                        +(weighted(q, P+step)[1]-weighted(q, P-step)[1]))/(2*step)
        assert abs(weighted_div) < 2e-7
        # Worked Jacobi triple (Q,P,QP), independent of the general 2D proof.
        derivative_factor = factor/length
        terms = [-factor*factor, factor*(derivative_factor*q+factor), -factor*q*derivative_factor]
        assert abs(sum(terms)) < 2e-12
        count += 1
    for t in [0, .25, .5, 1, 2]:
        correct = [t, exp(t)]
        naive = [log(1+t), 1+t]
        for x, P in [correct, naive]:
            assert abs(exp(-2*x)*P*P/2-.5) < 2e-15
            assert abs(exp(-x)*P-1) < 2e-15
        assert abs(exp(-2*naive[0])*naive[1]-1/(1+t)) < 1e-14
    return count


def lie_matrix(angular):
    x, y, z = angular
    return [[0, z, -y], [-z, 0, x], [y, -x, 0]]


def reduced_checks():
    count = 0
    for position, momentum in product(product(map(Q, [-1, 0, 1]), repeat=3), repeat=2):
        angular = cross(position, momentum)
        matrix = lie_matrix(angular)
        assert [dot(row, angular) for row in matrix] == [0, 0, 0]
        for i in range(3):
            for j in range(3):
                ei = [Q(int(k == i)) for k in range(3)]
                ej = [Q(int(k == j)) for k in range(3)]
                bracket = dot(cross(momentum, ei), cross(ej, position))-dot(cross(ei, position), cross(momentum, ej))
                assert bracket == matrix[i][j]
        count += 1
    A = ([Q(1), Q(1), Q(0)], [Q(0), Q(1), Q(0)])
    B = ([Q(5, 7), Q(-1, 5), Q(0)], [Q(0), Q(7, 5), Q(0)])
    torques = []
    for position, momentum in [A, B]:
        assert cross(position, momentum) == [0, 0, 1]
        assert dot(momentum, momentum)/2+position[1]**2/2+2*position[2]**2 == 1
        force = [0, -position[1], -4*position[2]]
        torques.append(cross(position, force))
    assert torques == [[0, 0, -1], [0, 0, Q(1, 7)]]
    zero_torque = cross([Q(1)]*3, [Q(0), Q(-1), Q(-4)])
    assert zero_torque == [-3, 4, -1] and dot(zero_torque, zero_torque) == 26
    step = 1e-4
    def zero_case_c(t):
        x = [1, cos(t), cos(2*t)]
        p = [0, -sin(t), -2*sin(2*t)]
        angular = cross(x, p)
        return dot(angular, angular)/2
    assert abs((zero_case_c(step)+zero_case_c(-step))/step**2-26) < 5e-6
    for t in [0, .25, .5, 1, 2]:
        reduced = [0, -sin(t), cos(t)]  # h=Lx, with the positive spatial bracket.
        rate = [dot(row, [1, 0, 0]) for row in lie_matrix(reduced)]
        assert max(abs(a-b) for a, b in zip(rate, [0, -cos(t), -sin(t)])) < 2e-15
        assert abs(dot(reduced, reduced)/2-.5) < 2e-15
        for x, p in [A, B]:
            y = float(x[1])*cos(t)+float(p[1])*sin(t)
            py = float(p[1])*cos(t)-float(x[1])*sin(t)
            assert abs((y*y+py*py)/2-1) < 2e-15
            def c_at(time):
                lz = float(x[0])*(float(p[1])*cos(time)-float(x[1])*sin(time))
                return lz*lz/2
            lz = float(x[0])*py
            numeric_rate = (c_at(t+1e-5)-c_at(t-1e-5))/(2e-5)
            assert abs(numeric_rate-lz*(-float(x[0])*y)) < 2e-8
    return count


if __name__ == '__main__':
    result = {'legendreStates': legendre_checks(), 'nonlinearChartStates': nonlinear_checks(),
              'canonicalAngularStates': reduced_checks(), 'clockExamples': 5,
              'sameAngularAndEnergyCounterexamples': 2, 'zeroLeafCounterexample': 1}
    if '--json' in sys.argv:
        print(json.dumps(result))
    else:
        print(result)
        for quartic in [0, .5, 1]:
            u = speed(sqrt(18), 1, quartic)
            print('lambda, speed at fixed P, Hamiltonian at fixed P, energy at fixed speed:',
                  quartic, u, hamiltonian(1, quartic, 1, 3, 3), 1+3*quartic)
