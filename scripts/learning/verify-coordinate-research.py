"""Independent polar Hessian and double-pendulum small-oscillation checks.

Run with Python 3; only the standard library is required. These are companion
examples, rather than additional controls in the instantaneous geometry labs.
"""
from fractions import Fraction as Q
from math import cos, sin, pi, sqrt


def polar_hessian(mass, stiffness, offset, radius, angle):
    if min(mass, stiffness, offset, radius) <= 0:
        raise ValueError('Positive mass, stiffness, offset and radius required.')
    gradient = [stiffness * (radius - offset * cos(angle)),
                stiffness * offset * radius * sin(angle)]
    ordinary = [[stiffness, stiffness * offset * sin(angle)],
                [stiffness * offset * sin(angle), stiffness * offset * radius * cos(angle)]]
    # Nonzero connection coefficients: Gamma^r_tt=-r, Gamma^t_rt=1/r.
    covariant = [[ordinary[0][0], ordinary[0][1] - gradient[1] / radius],
                 [ordinary[1][0] - gradient[1] / radius,
                  ordinary[1][1] + radius * gradient[0]]]
    jacobian = [[cos(angle), -radius * sin(angle)],
                [sin(angle), radius * cos(angle)]]
    metric = [[mass * sum(row[i] * row[j] for row in jacobian) for j in range(2)]
              for i in range(2)]
    return gradient, ordinary, covariant, jacobian, metric


def polar_checks():
    count = 0
    for mass in [.5, 1, 2]:
        for stiffness in [.5, 1, 3]:
            for offset in [.5, 1]:
                for radius in [.5, 1, 2]:
                    for angle in [0, pi/3, pi/2, pi]:
                        grad, ordinary, covariant, jacobian, metric = polar_hessian(
                            mass, stiffness, offset, radius, angle)
                        # Differentiate the composed Cartesian scalar numerically.
                        def potential(r, t):
                            x, y = r * cos(t), r * sin(t)
                            return stiffness * ((x-offset)**2+y*y) / 2
                        h = 1e-4
                        center = potential(radius, angle)
                        rr = (potential(radius+h, angle)-2*center+potential(radius-h, angle))/h**2
                        tt = (potential(radius, angle+h)-2*center+potential(radius, angle-h))/h**2
                        rt = (potential(radius+h, angle+h)-potential(radius+h, angle-h)
                              -potential(radius-h, angle+h)+potential(radius-h, angle-h))/(4*h*h)
                        assert max(abs(a-b) for a, b in zip(
                            [rr, rt, tt], [ordinary[0][0], ordinary[0][1], ordinary[1][1]])) < 4e-6
                        for i in range(2):
                            for j in range(2):
                                pullback = stiffness * sum(row[i]*row[j] for row in jacobian)
                                assert abs(covariant[i][j]-pullback) < 3e-12
                                assert abs(covariant[i][j]-stiffness*metric[i][j]/mass) < 3e-12
                        count += 1
    for angle in [0, pi/2, pi]:
        _, hessian, covariant, _, metric = polar_hessian(1, 1, 1, 1, angle)
        print('angle/pi, ordinary angular curvature, covariant curvature:',
              round(angle/pi, 6), round(hessian[1][1]/metric[1][1], 6),
              round(covariant[1][1]/metric[1][1], 6))
    print(f'{count} polar Hessian states passed; composed-potential derivatives and Cartesian pullbacks checked')


def transpose(a):
    return [list(row) for row in zip(*a)]


def multiply(a, b):
    return [[sum(a[i][k]*b[k][j] for k in range(2)) for j in range(2)] for i in range(2)]


def determinant(a):
    return a[0][0]*a[1][1]-a[0][1]*a[1][0]


def inverse(a):
    d = determinant(a)
    return [[a[1][1]/d, -a[0][1]/d], [-a[1][0]/d, a[0][0]/d]]


def quadratic(a, x):
    return sum(x[i]*a[i][j]*x[j] for i in range(2) for j in range(2))


def pencil(mass, stiffness):
    a = determinant(mass)
    b = -(stiffness[0][0]*mass[1][1]+stiffness[1][1]*mass[0][0]
          -stiffness[0][1]*mass[1][0]-stiffness[1][0]*mass[0][1])
    return [Q(1), b/a, determinant(stiffness)/a]


def mode_checks():
    count = 0
    for m1, m2 in [(1, 1), (2, 1), (1, 3)]:
        for l1, l2 in [(1, 1), (1, 2), (2, 1)]:
            for gravity in [1, 10]:
                mass = [[Q((m1+m2)*l1*l1), Q(m2*l1*l2)],
                        [Q(m2*l1*l2), Q(m2*l2*l2)]]
                stiffness = [[Q((m1+m2)*gravity*l1), Q(0)], [Q(0), Q(m2*gravity*l2)]]
                for a in [[[1, 0], [1, 1]], [[2, 0], [1, 1]], [[1, 1], [0, 1]]]:
                    a = [[Q(v) for v in row] for row in a]
                    relative_mass = multiply(multiply(transpose(a), mass), a)
                    relative_stiffness = multiply(multiply(transpose(a), stiffness), a)
                    assert pencil(mass, stiffness) == pencil(relative_mass, relative_stiffness)
                    dynamical = multiply(inverse(mass), stiffness)
                    assert multiply(inverse(relative_mass), relative_stiffness) == multiply(multiply(inverse(a), dynamical), a)
                    x = [[Q(1)], [Q(-2)]]
                    u = multiply(inverse(a), [[Q(1), Q(0)], [Q(-2), Q(0)]])
                    assert quadratic(mass, [row[0] for row in x]) == quadratic(relative_mass, [row[0] for row in u])
                    assert quadratic(stiffness, [row[0] for row in x]) == quadratic(relative_stiffness, [row[0] for row in u])
                    count += 1
    for sign in [-1, 1]:
        eigenvalue = (3+sign*sqrt(3))/2
        vector = [1, -sign*sqrt(3)]
        mass = [[3, 1], [1, 1]]
        stiffness = [[3, 0], [0, 1]]
        residual = [sum((stiffness[i][j]-eigenvalue*mass[i][j])*vector[j] for j in range(2)) for i in range(2)]
        assert max(map(abs, residual)) < 2e-15
        print('frequency squared, frequency, absolute mode:', eigenvalue, sqrt(eigenvalue), vector)
    print(f'{count} exact mass/stiffness coordinate pairs passed; physical quadratic forms and spectral pencils agree')


def nonlinear_rate(state):
    a, b, v, w = state
    coupling, s = cos(a-b), sin(a-b)
    right = [-s*w*w-3*sin(a), s*v*v-sin(b)]
    d = 3-coupling*coupling
    return [v, w, (right[0]-coupling*right[1])/d,
            (3*right[1]-coupling*right[0])/d]


def nonlinear_trajectory(amplitude, sign, step_count):
    state = [amplitude, -sign*sqrt(3)*amplitude, 0., 0.]
    h = 2/step_count
    for _ in range(step_count):
        k1 = nonlinear_rate(state)
        k2 = nonlinear_rate([x+h*k/2 for x, k in zip(state, k1)])
        k3 = nonlinear_rate([x+h*k/2 for x, k in zip(state, k2)])
        k4 = nonlinear_rate([x+h*k for x, k in zip(state, k3)])
        state = [x+h*(a+2*b+2*c+d)/6 for x, a, b, c, d in zip(state, k1, k2, k3, k4)]
    return state


def nonlinear_checks():
    for sign in [-1, 1]:
        errors = []
        frequency = sqrt((3+sign*sqrt(3))/2)
        for amplitude in [.04, .02, .01]:
            coarse = nonlinear_trajectory(amplitude, sign, 1000)
            state = nonlinear_trajectory(amplitude, sign, 2000)
            assert max(abs(a-b) for a, b in zip(coarse, state)) < 1e-10
            target = [amplitude*cos(2*frequency), -sign*sqrt(3)*amplitude*cos(2*frequency)]
            errors.append(sqrt(sum((a-b)**2 for a, b in zip(state[:2], target))))
        ratios = [errors[i]/errors[i+1] for i in range(2)]
        assert all(7.5 < ratio < 8.5 for ratio in ratios)
        print('mode sign, fixed-time angular errors, halving ratios:', sign, errors, ratios)
    print('6 nonlinear trajectories passed with 6 step-halving comparisons at fixed time 2 s')


if __name__ == '__main__':
    polar_checks()
    mode_checks()
    nonlinear_checks()
