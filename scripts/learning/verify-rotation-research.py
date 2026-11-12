"""Independent checks for the two rotation research extensions.

Run from the repository root with Python 3. The default checks use only the
standard library. --figures additionally requires NumPy and Matplotlib;
--independent-ode additionally requires SciPy. No production code is imported.
"""

import argparse
from fractions import Fraction as Q
from math import cosh, sqrt, tanh


def geometry(masses, lengths):
    points = []
    for j, (mass, length) in enumerate(zip(masses, lengths)):
        for sign in [-1, 1]:
            position = [Q(0)] * 3
            position[j] = sign * Q(length)
            points.append((Q(mass), position))
    return points


def second_moment(points):
    return [[sum(m * r[i] * r[j] for m, r in points)
             for j in range(3)] for i in range(3)]


def inertia(points):
    s = second_moment(points)
    trace = sum(s[j][j] for j in range(3))
    return [[(trace if i == j else 0) - s[i][j]
             for j in range(3)] for i in range(3)]


def energy(tensor, velocity):
    return sum(velocity[i] * tensor[i][j] * velocity[j]
               for i in range(3) for j in range(3)) / 2


def shape_checks():
    a = geometry([4, 1, 1], [Q(1, 2), 1, 1])
    b = geometry([1, 1, 4], [1, 1, Q(1, 2)])
    expected = [[Q(4 if i == j else 0) for j in range(3)]
                for i in range(3)]
    assert inertia(a) == inertia(b) == expected
    assert sum(m for m, _ in a) == sum(m for m, _ in b) == 12
    assert energy(expected, [1, 1, 0]) == 4
    fourth = [sum(m * r[0] ** 4 for m, r in p) for p in [a, b]]
    assert fourth == [Q(1, 2), Q(2)]
    # Recompute point sums about O=(1,0,0), not just a tensor formula.
    shifted = [[(m, [r[0] - 1, r[1], r[2]]) for m, r in p]
               for p in [a, b]]
    ip = [[Q(4), Q(0), Q(0)], [Q(0), Q(16), Q(0)],
          [Q(0), Q(0), Q(16)]]
    assert inertia(shifted[0]) == inertia(shifted[1]) == ip
    assert energy(ip, [1, 1, 0]) == 10

    # Six energies reconstruct a general symmetric tensor, not its geometry.
    original = [[Q(26), Q(0), Q(-3)], [Q(0), Q(41, 2), Q(0)],
                [Q(-3), Q(0), Q(21, 2)]]
    axes = [[Q(int(i == j)) for i in range(3)] for j in range(3)]
    diagonal_energy = [energy(original, w) for w in axes]
    recovered = [[Q(0) for _ in range(3)] for _ in range(3)]
    for i in range(3):
        recovered[i][i] = 2 * diagonal_energy[i]
        for j in range(i):
            w = [axes[i][k] + axes[j][k] for k in range(3)]
            recovered[i][j] = recovered[j][i] = (
                energy(original, w) - diagonal_energy[i] - diagonal_energy[j])
    assert recovered == original
    impossible = [Q(1), Q(1), Q(3)]
    s_diag = [sum(impossible) / 2 - x for x in impossible]
    assert s_diag == [Q(3, 2), Q(3, 2), Q(-1, 2)]
    print('Same mass/tensor, different fourth moments:', *fourth)
    print('COM/pivot energies (J): 4, 10; six-probe reconstruction passed')
    return a, b


def separatrix(t, omega=1.0):
    """Body components and their analytic time derivative; I=(3,4,5)."""
    rate = omega / sqrt(15)
    u = rate * t
    sech = 1 / cosh(u)
    v = tanh(u)
    w = [sqrt(2 / 3) * omega * sech, omega * v,
         sqrt(2 / 5) * omega * sech]
    derivative = [-rate * w[0] * v, omega * rate * sech ** 2,
                  -rate * w[2] * v]
    return w, derivative


def euler_rhs(w):
    x, y, z = w
    return [-y * z / 3, z * x / 2, -x * y / 5]


def invariants(w):
    return (sum(i * x * x / 2 for i, x in zip([3, 4, 5], w)),
            sum((i * x) ** 2 for i, x in zip([3, 4, 5], w)))


def separatrix_checks():
    count = 0
    for omega in [.5, 1, 2]:
        for j in range(-80, 81):
            t = j * sqrt(15) / (20 * omega)
            w, derivative = separatrix(t, omega)
            rhs = euler_rhs(w)
            assert max(abs(a - b) for a, b in zip(derivative, rhs)) < 2e-14
            k, l2 = invariants(w)
            assert abs(k - 2 * omega ** 2) < 2e-14
            assert abs(l2 - 16 * omega ** 2) < 2e-13
            count += 1
    t = sqrt(15)
    w, _ = separatrix(t)
    wrong, wrong_derivative = separatrix(1.1 * t)
    error = max(abs(a - b) for a, b in zip(w, wrong))
    assert .038 < error < .040
    residual = max(abs(1.1 * a - b)
                   for a, b in zip(wrong_derivative, euler_rhs(wrong)))
    assert residual > .009
    assert separatrix(0)[0] == separatrix(1.1 * 0)[0]
    assert abs(invariants(wrong)[0] - 2) < 1e-14
    assert abs(invariants(wrong)[1] - 16) < 1e-13
    print(f'{count} analytic Euler/invariant samples passed')
    print('Wrong clock at t=sqrt(15): max angular-velocity error, ODE residual:',
          error, residual)


def independent_ode():
    from scipy.integrate import solve_ivp
    # This integrates body components directly. It does not validate orientation.
    for omega in [.5, 1, 2]:
        end = 2 * sqrt(15) / omega
        solution = solve_ivp(lambda t, w: euler_rhs(w), [0, end],
                             separatrix(0, omega)[0], method='DOP853',
                             rtol=2e-13, atol=2e-14, dense_output=True)
        assert solution.success
        for j in range(41):
            t = end * j / 40
            assert max(abs(a - b) for a, b in zip(
                solution.sol(t), separatrix(t, omega)[0])) < 2e-11
    print('Independent DOP853: 3 trajectories / 123 body-component states passed')


def figures():
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'svg.fonttype': 'path', 'axes.spines.top': False,
                         'axes.spines.right': False})
    destination = Path('docs/online-learning/physics-math-for-ai/assets')
    destination.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        target = destination / name
        fig.savefig(target, metadata={'Date': None}, facecolor='white')
        target.write_text('\n'.join(line.rstrip() for line in
                                   target.read_text().splitlines()) + '\n')
        preview = Path('.runtime/learning') / name.replace('.svg', '.png')
        preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview, dpi=150, facecolor='white')
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(5.2, 7.2), layout='constrained')
    for ax, (masses, lengths, name) in zip(axes, [
        ([4, 1, 1], [.5, 1, 1], 'Shape A'),
        ([1, 1, 4], [1, 1, .5], 'Shape B'),
    ]):
        # xz projection: the y-axis pair overlaps at (0,0), explicitly labeled.
        ax.scatter([-lengths[0], lengths[0]], [0, 0],
                   s=80 * masses[0], color='#2563eb')
        ax.scatter([0, 0], [-lengths[2], lengths[2]],
                   s=80 * masses[2], color='#ea580c')
        ax.scatter([0], [0], s=100, facecolors='none', edgecolors='#64748b')
        ax.annotate('y pair overlaps here', (0, 0), (.08, .18), fontsize=10)
        ax.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), xlabel='x (m)',
               ylabel='z (m)', title=name + ': M = 12 kg, I = 4 Id kg m²')
        ax.set_aspect('equal')
        ax.grid(alpha=.15)
    save(fig, 'rotation-shape-nonidentifiability.svg')

    u = np.linspace(0, 3, 301)
    exact = np.array([separatrix(x * sqrt(15))[0] for x in u])
    wrong = np.array([separatrix(1.1 * x * sqrt(15))[0] for x in u])
    fig, axes = plt.subplots(2, 1, figsize=(5.2, 6.4), layout='constrained')
    axes[0].plot(u, exact[:, 1], color='#2563eb', label='Correct clock')
    axes[0].plot(u, wrong[:, 1], '--', color='#ea580c', label='Clock × 1.1')
    axes[0].set(ylabel='Body ω₂ (rad/s)', title='Same initial state, different clock')
    axes[0].legend(loc='lower right')
    error = np.max(np.abs(wrong - exact), axis=1)
    axes[1].plot(u, error, color='#2563eb', label='max component error')
    axes[1].set(xlabel='λt (dimensionless)', ylabel='Error (rad/s)',
                title='Same K = 2 J\nSame |L|² = 16 kg² m⁴/s²')
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.15)
    save(fig, 'rotation-invariant-wrong-clock.svg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--independent-ode', action='store_true')
    parser.add_argument('--figures', action='store_true')
    args = parser.parse_args()
    shape_checks()
    separatrix_checks()
    if args.independent_ode:
        independent_ode()
    if args.figures:
        figures()
