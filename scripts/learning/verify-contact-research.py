"""Exact, independent mass-metric and free-wheel research checks.

Run from the repository root with Python 3. --figures additionally requires
NumPy and Matplotlib. This script does not import the production evaluator.
"""
import argparse
from fractions import Fraction as Q
from itertools import product


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def mv(a, v):
    return [dot(row, v) for row in a]


def mm(a, b):
    return [[dot(row, col) for col in zip(*b)] for row in a]


def transpose(a):
    return [list(col) for col in zip(*a)]


def kinetic(mass_diag, v):
    return dot(mass_diag, [x * x for x in v]) / 2


def operators(mass, a, h, restitution):
    inertia = mass * (a * a + h * h) / 3
    mass_diag = [mass, mass, inertia]
    j = [Q(0), Q(1), a]
    w = sum(x * x / m for x, m in zip(j, mass_diag))
    normal = [x / m / w for x, m in zip(j, mass_diag)]
    identity = [[Q(int(i == k)) for k in range(3)] for i in range(3)]
    projection = [[identity[i][k] - normal[i] * j[k]
                   for k in range(3)] for i in range(3)]
    reflection = [[identity[i][k] - 2 * normal[i] * j[k]
                   for k in range(3)] for i in range(3)]
    update = [[identity[i][k] - (1 + restitution) * normal[i] * j[k]
               for k in range(3)] for i in range(3)]
    return mass_diag, j, w, normal, identity, projection, reflection, update


def restitution_error_bound(delta, noise, restitution):
    if not 0 <= noise < delta:
        raise ValueError('Incoming speed must be separated from zero.')
    return noise * (1 + restitution) / (delta - noise)


def impact_checks():
    cases = 0
    for mass, a, h in product(map(Q, [1, 2, 4]), [Q(0), Q(1, 2), Q(3, 2)],
                              [Q(1, 2), Q(1), Q(2)]):
        d, j, w, n, identity, p, r, _ = operators(mass, a, h, Q(0))
        m = [[d[i] if i == k else Q(0) for k in range(3)] for i in range(3)]
        assert mm(p, p) == p
        assert mm(transpose(p), m) == mm(m, p)
        assert mm(r, r) == identity
        assert mm(mm(transpose(r), m), r) == m
        for e in [Q(0), Q(1, 2), Q(1)]:
            *_, update = operators(mass, a, h, e)
            for velocity in [[Q(0), Q(-1), Q(0)], [Q(1), Q(1, 2), Q(-1)],
                             [Q(-1), Q(0), Q(1)], [Q(0), -a, Q(1)]]:
                c = dot(j, velocity)
                after = mv(update, velocity) if c < 0 else velocity[:]
                impulse = -(1 + e) * min(c, Q(0)) / w
                assert impulse >= 0
                assert dot(j, after) == (-e * c if c < 0 else c)
                loss = kinetic(d, velocity) - kinetic(d, after)
                assert loss == ((1 - e * e) * c * c / (2 * w) if c < 0 else 0)
                if e == 0:
                    # Feasible competitors differ by a tangent and a nonnegative
                    # normal component; compare the projection objective exactly.
                    optimum = kinetic(d, [x - y for x, y in zip(after, velocity)])
                    for tangent_x, tangent, opening in product(
                            [Q(-1), Q(0), Q(1)], [Q(-1), Q(0), Q(1)],
                            [Q(0), Q(1, 2), Q(1)]):
                        offset = [tangent_x, -a * tangent, tangent]
                        candidate = [x + y + opening * z for x, y, z in zip(after, offset, n)]
                        assert dot(j, candidate) >= 0
                        cost = kinetic(d, [x - y for x, y in zip(candidate, velocity)])
                        assert cost >= optimum
                        if tangent_x == tangent == opening == 0:
                            assert cost == optimum
                cases += 1
    e, noise = Q(1, 2), Q(1, 1000)
    for delta in [Q(1, 10), Q(1, 100)]:
        bound = restitution_error_bound(delta, noise, e)
        for incoming_error, outgoing_error in product([-noise, Q(0), noise], repeat=2):
            estimate = (e * delta + outgoing_error) / (delta - incoming_error)
            assert abs(estimate - e) <= bound
        print('Incoming speed, restitution error bound:', delta, bound)
    try:
        restitution_error_bound(noise, noise, e)
    except ValueError:
        pass
    else:
        raise AssertionError('Grazing interval needs rejection')
    print(f'27 geometries: projection/reflection identities; {cases} restitution states passed')


def free_wheel(mass, radius, kappa, mu, v0, omega0, time):
    """Zero external force/torque, g=10, fixed height, clockwise omega."""
    effective_mass = mass * kappa / (1 + kappa)
    rolling_speed = (v0 + kappa * radius * omega0) / (1 + kappa)
    slip0 = v0 - radius * omega0
    magnitude = abs(slip0)
    slope = mu * mass * 10 / effective_mass
    event = magnitude / slope if slope > 0 and magnitude > 0 else None
    slip = (1 if slip0 > 0 else -1 if slip0 < 0 else 0) * max(magnitude - slope * time, Q(0))
    duration = min(time, event) if event is not None else time
    signed_area = (1 if slip0 > 0 else -1 if slip0 < 0 else 0) * (
        magnitude * duration - slope * duration * duration / 2)
    velocity = rolling_speed + kappa * slip / (1 + kappa)
    omega = (rolling_speed - slip / (1 + kappa)) / radius
    position = rolling_speed * time + kappa * signed_area / (1 + kappa)
    angle = (rolling_speed * time - signed_area / (1 + kappa)) / radius
    heat = effective_mass * (slip0 * slip0 - slip * slip) / 2
    energy = mass * velocity * velocity / 2 + kappa * mass * radius * radius * omega * omega / 2
    rolling_energy = mass * (1 + kappa) * rolling_speed * rolling_speed / 2
    assert energy == rolling_energy + effective_mass * slip * slip / 2
    assert mass * v0 * v0 / 2 + kappa * mass * radius * radius * omega0 * omega0 / 2 == energy + heat
    assert heat >= 0
    return dict(velocity=velocity, omega=omega, position=position, angle=angle,
                heat=heat, energy=energy, slip=slip, event=event)


def wheel_checks():
    count = 0
    for m, r, k, mu, v, omega, t in product(
            map(Q, [1, 2]), [Q(1, 2), Q(1)], [Q(1, 2), Q(1)],
            [Q(0), Q(1, 10), Q(1, 5)], map(Q, [-1, 0, 1]), map(Q, [-1, 0, 1]),
            [Q(0), Q(3, 10), Q(2, 5), Q(1), Q(10)]):
        free_wheel(m, r, k, mu, v, omega, t)
        count += 1
    for mu in [Q(0), Q(1, 10), Q(1, 5), Q(1, 2)]:
        s = free_wheel(Q(1), Q(1), Q(1, 2), mu, Q(1), Q(0), Q(1))
        assert s['heat'] == (Q(1, 6) if mu > 0 else 0)
        print('Native mu, event, V, Omega, heat:', mu, s['event'], s['velocity'], s['omega'], s['heat'])
    for t, expected in [(Q(3, 10), Q(33, 200)), (Q(2, 5), Q(1, 6))]:
        assert free_wheel(Q(1), Q(1), Q(1, 2), Q(1, 10), Q(1), Q(0), t)['heat'] == expected
    print(f'{count} exact free-wheel mode and energy checks passed')


def figures():
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 12, 'svg.fonttype': 'path',
                         'svg.hashsalt': 'contact-research-foundation'})
    root = Path('docs/online-learning/physics-math-for-ai/assets')
    root.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        p = root / (name + '.svg')
        fig.savefig(p, metadata={'Date': None}, facecolor='white')
        p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines()) + '\n')
        fig.savefig(Path('.runtime/learning') / (name + '.png'), dpi=120, facecolor='white')
        plt.close(fig)

    c = np.linspace(-1, 1, 201)
    fig, axes = plt.subplots(2, 1, figsize=(5.2, 7), layout='constrained')
    axes[0].plot(c, -1.5 * np.minimum(c, 0) / 2.5, color='#2563eb')
    axes[0].set(xlabel='Incoming normal speed (m/s)', ylabel='Impulse (N s)',
                title='Continuous impulse\nDerivative changes at zero')
    delta = np.geomspace(.0011, .2, 201)
    axes[1].plot(delta, .0015 / (delta - .001), color='#ea580c')
    axes[1].set(xscale='log', yscale='log', xlabel='Closing speed delta (m/s)',
                ylabel='Restitution error bound', title='Fixed speed error = 0.001 m/s')
    for ax in axes:
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'contact-grazing-conditioning')
    fig, axes = plt.subplots(2, 1, figsize=(5.2, 7), layout='constrained')
    times = np.linspace(0, 2, 401)
    for mu, color in [(0, '#64748b'), (.02, '#ea580c'), (.1, '#2563eb')]:
        heat = (1 - np.maximum(1 - 30 * mu * times, 0) ** 2) / 6
        axes[0].plot(times, heat, label=f'mu = {mu}', color=color)
    axes[0].set(xlabel='Time (s)', ylabel='Heat (J)', title='Same eventual loss\nfor every mu > 0')
    axes[0].legend(loc='lower right')
    mu = np.geomspace(.001, 1, 201)
    axes[1].loglog(mu, 1 / (30 * mu), color='#2563eb')
    axes[1].set(xlabel='mu (dimensionless)', ylabel='Rolling onset time (s)',
                title='Small positive friction\nLong waiting time')
    for ax in axes:
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'contact-small-friction-limits')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures', action='store_true')
    args = parser.parse_args()
    impact_checks()
    wheel_checks()
    if args.figures:
        figures()
