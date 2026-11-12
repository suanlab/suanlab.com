"""Independent kinetic-candidate and pendulum mass-scale experiments.

Python standard library only. --json emits reproducible trajectory references.
The pendulum equations are obtained by Cartesian virtual-work projection,
rather than importing the website's expanded Lagrange equations.
"""
from fractions import Fraction as Q
from itertools import product
from math import sin, cos, pi
import json
import sys


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def solve(matrix, force):
    determinant = matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0]
    return [(matrix[1][1]*force[0]-matrix[0][1]*force[1])/determinant,
            (matrix[0][0]*force[1]-matrix[1][0]*force[0])/determinant]


def kinetic(mass, quartic, velocity):
    speed2 = dot(velocity, velocity)
    lagrangian = mass*speed2/2+quartic*speed2*speed2/4
    momentum = [(mass+quartic*speed2)*v for v in velocity]
    hessian = [[(mass+quartic*speed2)*int(i == j)+2*quartic*velocity[i]*velocity[j]
                for j in range(2)] for i in range(2)]
    energy = dot(momentum, velocity)-lagrangian
    return lagrangian, momentum, hessian, energy


def derivative(function, point, axis):
    # Exact for the scalar polynomial degrees used here (at most four).
    step = Q(1, 7)
    values = []
    for shift in [-2, -1, 1, 2]:
        probe = list(point)
        probe[axis] += shift*step
        values.append(function(probe))
    return (values[0]-8*values[1]+8*values[2]-values[3])/(12*step)


def kinetic_checks():
    count = 0
    for mass, quartic, velocity, force in product(
            map(Q, [1, 2]), map(Q, ['0', '1/2', '1']),
            [[Q(0), Q(0)], [Q(1), Q(0)], [Q(1), Q(1)], [Q(-1), Q(2)]],
            [[Q(0), Q(0)], [Q(1), Q(0)], [Q(0), Q(1)]]):
        lagrangian, momentum, hessian, energy = kinetic(mass, quartic, velocity)
        for i in range(2):
            assert derivative(lambda v: kinetic(mass, quartic, v)[0], velocity, i) == momentum[i]
            for j in range(2):
                assert derivative(lambda v: kinetic(mass, quartic, v)[1][i], velocity, j) == hessian[i][j]
        assert hessian[0][0] > 0
        assert hessian[0][0]*hessian[1][1]-hessian[0][1]*hessian[1][0] > 0
        acceleration = solve(hessian, force)
        assert [dot(row, acceleration) for row in hessian] == force
        energy_gradient = [derivative(lambda v: kinetic(mass, quartic, v)[3], velocity, i) for i in range(2)]
        assert dot(energy_gradient, acceleration) == dot(force, velocity)
        if force == [0, 0]:
            assert acceleration == [0, 0]
        count += 1
    boosts = 0
    for mass, quartic, boost in product(map(Q, [1, 2]), map(Q, ['0', '1/2', '1']), map(Q, [1, 2])):
        before = kinetic(mass, quartic, [Q(0), Q(0)])[2]
        after = kinetic(mass, quartic, [boost, Q(0)])[2]
        difference = [[after[i][j]-before[i][j] for j in range(2)] for i in range(2)]
        assert difference == [[3*quartic*boost*boost, 0], [0, quartic*boost*boost]]
        boosts += 1
    return count, boosts


def pendulum(masses, lengths, angles, rates, force=(0, 0)):
    m1, m2 = masses
    l1, l2 = lengths
    (sa, ca), (sb, cb) = angles
    wa, wb = rates
    gravity = 10
    jacobian = [[l1*ca, 0], [l1*sa, 0], [l1*ca, l2*cb], [l1*sa, l2*sb]]
    bias = [-l1*wa*wa*sa, l1*wa*wa*ca,
            -l1*wa*wa*sa-l2*wb*wb*sb, l1*wa*wa*ca+l2*wb*wb*cb]
    weights = [m1, m1, m2, m2]
    applied = [0, -m1*gravity, force[0], force[1]-m2*gravity]
    matrix = [[sum(weights[k]*jacobian[k][i]*jacobian[k][j] for k in range(4))
               for j in range(2)] for i in range(2)]
    right = [sum(jacobian[k][i]*(applied[k]-weights[k]*bias[k]) for k in range(4)) for i in range(2)]
    acceleration = solve(matrix, right)
    cartesian_acceleration = [dot(row, acceleration)+b for row, b in zip(jacobian, bias)]
    velocity = [dot(row, rates) for row in jacobian]
    kinetic_energy = sum(w*v*v/2 for w, v in zip(weights, velocity))
    potential = gravity*((m1+m2)*l1*(1-ca)+m2*l2*(1-cb))
    rod2 = [m2*cartesian_acceleration[2]-force[0],
            m2*(cartesian_acceleration[3]+gravity)-force[1]]
    rod1 = [m1*cartesian_acceleration[0]+rod2[0],
            m1*(cartesian_acceleration[1]+gravity)+rod2[1]]
    energy_rate = sum(w*v*a for w, v, a in zip(weights, velocity, cartesian_acceleration))
    energy_rate += gravity*(m1*velocity[1]+m2*velocity[3])
    tip_generalized_force = [jacobian[2][i]*force[0]+jacobian[3][i]*force[1] for i in range(2)]
    return {'matrix': matrix, 'acceleration': acceleration,
            'cartesianAcceleration': cartesian_acceleration, 'velocity': velocity,
            'momentum': [dot(row, rates) for row in matrix],
            'energy': kinetic_energy+potential, 'rod1': rod1, 'rod2': rod2,
            'energyRate': energy_rate, 'tipGeneralizedForce': tip_generalized_force}


def scale_checks():
    pairs = [(Q(0), Q(1)), (Q(1), Q(0)), (Q(3, 5), Q(4, 5)), (Q(-4, 5), Q(3, 5))]
    angles = [(pairs[0], pairs[0]), (pairs[1], pairs[0]), (pairs[2], pairs[3]), (pairs[3], pairs[2])]
    count = forced_count = 0
    for masses, lengths, pose, rates in product(
            [(Q(1), Q(1)), (Q(1), Q(2)), (Q(2), Q(1))],
            [(Q(1), Q(1)), (Q(1, 2), Q(2))], angles,
            [(Q(0), Q(0)), (Q(1), Q(-1)), (Q(1, 2), Q(1, 3))]):
        base = pendulum(masses, lengths, pose, rates)
        assert dot(base['rod1'], [pose[0][1], pose[0][0]]) == 0
        assert dot(base['rod2'], [pose[1][1], pose[1][0]]) == 0
        constraint_power = dot([a-b for a, b in zip(base['rod1'], base['rod2'])], base['velocity'][:2])
        constraint_power += dot(base['rod2'], base['velocity'][2:])
        assert constraint_power == 0
        for scale in map(Q, ['1/2', '3/2', '2']):
            scaled_masses = tuple(scale*m for m in masses)
            scaled = pendulum(scaled_masses, lengths, pose, rates)
            for field in ['acceleration', 'cartesianAcceleration', 'velocity']:
                assert scaled[field] == base[field]
            for field in ['momentum', 'rod1', 'rod2']:
                assert scaled[field] == [scale*x for x in base[field]]
            assert scaled['energy'] == scale*base['energy']
            assert scaled['energyRate'] == base['energyRate'] == 0
            for i in range(2):
                assert scaled['matrix'][i] == [scale*x for x in base['matrix'][i]]
            count += 1
            for force in [(Q(1), Q(0)), (Q(0), Q(1))]:
                driven = pendulum(masses, lengths, pose, rates, force)
                driven_scaled = pendulum(scaled_masses, lengths, pose, rates, force)
                response = solve(base['matrix'], driven['tipGeneralizedForce'])
                assert [a-b for a, b in zip(driven_scaled['acceleration'], driven['acceleration'])] == [(1/scale-1)*x for x in response]
                assert driven['energyRate'] == dot(force, driven['velocity'][2:])
                assert driven_scaled['energyRate'] == dot(force, driven_scaled['velocity'][2:])
                forced_count += 1
    for scale in map(Q, ['1/2', '1', '2']):
        state = pendulum((scale, scale), (Q(1), Q(1)), (pairs[0], pairs[0]), (Q(0), Q(0)), (Q(1), Q(0)))
        assert state['acceleration'] == [0, 1/scale]
        normal_force = pendulum((scale, scale), (Q(1), Q(1)), (pairs[0], pairs[0]), (Q(0), Q(0)), (Q(0), Q(1)))
        assert normal_force['tipGeneralizedForce'] == [0, 0]
        assert normal_force['acceleration'] == [0, 0]
    return count, forced_count


def trajectory(scale):
    state = [pi/2, 0., 1., -1.]
    states = [list(state)]
    def rate(z):
        angles = [(sin(z[0]), cos(z[0])), (sin(z[1]), cos(z[1]))]
        result = pendulum((scale, scale), (1., 1.), angles, z[2:])
        return [z[2], z[3], *result['acceleration']]
    step = .01/8
    for _ in range(200):
        for _ in range(8):
            k1 = rate(state)
            k2 = rate([x+step*k/2 for x, k in zip(state, k1)])
            k3 = rate([x+step*k/2 for x, k in zip(state, k2)])
            k4 = rate([x+step*k for x, k in zip(state, k3)])
            state = [x+step*(a+2*b+2*c+d)/6 for x, a, b, c, d in zip(state, k1, k2, k3, k4)]
        states.append(list(state))
    return {'scale': scale, 'states': states}


def run():
    kinetic_count, boost_count = kinetic_checks()
    scale_count, forced_count = scale_checks()
    trajectories = [trajectory(c) for c in [.5, 1., 2.]]
    reference = trajectories[1]['states']
    maximum = max(abs(a-b) for row in trajectories for state, target in zip(row['states'], reference) for a, b in zip(state, target))
    assert maximum < 5e-12
    return {'kineticStates': kinetic_count, 'boostChecks': boost_count,
            'massScaleStates': scale_count, 'forcedScaleStates': forced_count,
            'trajectoryStatesPerScale': len(reference), 'maximumScaleGap': maximum,
            'trajectories': trajectories}


if __name__ == '__main__':
    result = run()
    if '--json' in sys.argv:
        print(json.dumps(result))
    else:
        for quartic in map(Q, ['0', '1/2', '1']):
            _, momentum, hessian, energy = kinetic(Q(1), quartic, [Q(1), Q(1)])
            print('quartic coefficient, energy, initial forced acceleration:', quartic, energy, solve(hessian, [Q(1), Q(0)]))
        print(f"{result['kineticStates']} exact kinetic-response states and {result['boostChecks']} boost Hessians passed")
        print(f"{result['massScaleStates']} exact mass-scale states and {result['forcedScaleStates']} calibrated-force comparisons passed")
        print('3 independent projection/RK4 trajectories passed at 201 times each; maximum scale gap:', result['maximumScaleGap'])
