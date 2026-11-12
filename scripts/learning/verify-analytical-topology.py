"""Independent punctured-plane periods and circle gauge-action checks.

Run from the repository root with Python 3. Only the standard library is used.
The examples are separate from the native linear-graph and 1D time-gauge labs.
"""
from fractions import Fraction as Q
from math import atan2, cos, sin, pi


def momentum(x, y, kappa):
    radius2 = x * x + y * y
    if radius2 == 0:
        raise ValueError('The origin is excluded from this configuration space.')
    return [-kappa * y / radius2, kappa * x / radius2]


def derivative(x, y, kappa):
    radius4 = (x * x + y * y) ** 2
    if radius4 == 0:
        raise ValueError('No derivative at the excluded origin.')
    return [[2 * kappa * x * y / radius4, kappa * (y * y - x * x) / radius4],
            [kappa * (y * y - x * x) / radius4, -2 * kappa * x * y / radius4]]


def loop_action(center, radius, kappa, winding, steps=1024):
    # Trapezoidal quadrature of p(q) dot q'(theta); no principal-angle subtraction.
    total = 0.0
    for j in range(steps):
        angle = 2 * pi * j / steps
        c, s = cos(winding * angle), sin(winding * angle)
        q = [center[0] + radius * c, center[1] + radius * s]
        velocity = [-radius * winding * s, radius * winding * c]
        p = momentum(*q, kappa)
        total += sum(a * b for a, b in zip(p, velocity))
    return total * 2 * pi / steps


def graph_checks():
    count = 0
    for kappa in [-1, 0, .5, 1]:
        for radius in [.5, 1, 2]:
            for winding in [-2, -1, 0, 1, 2]:
                expected = 2 * pi * kappa * winding
                assert abs(loop_action([0, 0], radius, kappa, winding) - expected) < 3e-12
                count += 1
        for winding in [-2, -1, 1, 2]:
            assert abs(loop_action([2, 0], .5, kappa, winding)) < 3e-12
            count += 1
        for x, y in [(.5, .25), (1, .5), (-1, 1), (2, -1)]:
            dp = derivative(x, y, kappa)
            step = 1e-5
            for j in range(2):
                plus, minus = [x, y], [x, y]
                plus[j] += step
                minus[j] -= step
                numeric = [(a - b) / (2 * step) for a, b in zip(
                    momentum(*plus, kappa), momentum(*minus, kappa))]
                assert max(abs(numeric[i] - dp[i][j]) for i in range(2)) < 2e-8
            assert dp[0][1] == dp[1][0]
            if x > 0:
                gradient = [kappa * (atan2(y, x + step) - atan2(y, x - step)) / (2 * step),
                            kappa * (atan2(y + step, x) - atan2(y - step, x)) / (2 * step)]
                assert max(abs(a - b) for a, b in zip(gradient, momentum(x, y, kappa))) < 2e-8
            # The graph also solves stationary HJ locally for this explicit U,
            # rather than for the free-particle Hamiltonian.
            for mass in [.5, 1, 2]:
                p = momentum(x, y, kappa)
                potential = -kappa * kappa / (2 * mass * (x * x + y * y))
                h = sum(a * a for a in p) / (2 * mass) + potential
                assert abs(h) < 2e-14
                transport = [sum(dp[i][j] * p[j] / mass for j in range(2)) for i in range(2)]
                force = [-kappa * kappa * q / (mass * (x * x + y * y) ** 2) for q in [x, y]]
                assert max(abs(a - b) for a, b in zip(transport, force)) < 2e-14
    print(f'{count} loop periods passed; derivatives, local actions and 48 Hamiltonian tangency states checked')
    for radius in [.5, 1, 2]:
        print('radius, winding loop, contractible loop:', radius,
              loop_action([0, 0], radius, 1, 1), loop_action([2, 0], .5, 1, 1))


def circle_state(alpha, winding):
    # SI example I=1 kg m^2, T=2*pi s, same physical endpoint angle.
    # Angular speed is winding rad/s; action/pi is exact in J s.
    alpha, winding = Q(alpha), Q(winding)
    momentum_value = winding + alpha
    energy = winding * winding / 2
    hamiltonian = (momentum_value - alpha) ** 2 / 2
    action_over_pi = winding * winding + 2 * alpha * winding
    assert energy == hamiltonian
    return momentum_value, energy, action_over_pi


def circle_checks():
    count = 0
    for alpha in [Q(-1), Q(0), Q(1, 2), Q(1)]:
        for winding in range(-4, 5):
            p, energy, action = circle_state(alpha, winding)
            assert p - alpha == winding
            assert action == (Q(winding) + alpha) ** 2 - alpha * alpha
            assert energy >= 0
            count += 1
    # Completing the square proves these are global minima over all integers;
    # enumeration is only an additional check over the displayed range.
    expected = {Q(0): [0], Q(1, 2): [-1, 0], Q(1): [-1]}
    for alpha, minimizers in expected.items():
        values = {n: circle_state(alpha, n)[2] for n in range(-4, 5)}
        assert [n for n in values if values[n] == min(values.values())] == minimizers
    for n in range(-2, 3):
        print('winding, base action/pi, shifted action/pi, energy:', n,
              circle_state(0, n)[2], circle_state(1, n)[2], circle_state(1, n)[1])
    print(f'{count} exact circle states passed; alpha=1/2 retains both minimizing winding sectors')


if __name__ == '__main__':
    graph_checks()
    circle_checks()
