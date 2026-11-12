"""Independent observation inversion and regularization-selection examples.

Default checks require Python 3 only. --independent-ode needs SciPy;
--figures needs NumPy and Matplotlib. Run from the repository root.
"""
import argparse
from itertools import product
from math import cosh, sinh, sqrt


def position(rate, x0, v0, time):
    return x0 + v0 * time if rate == 0 else (
        x0 * cosh(rate * time) + v0 * sinh(rate * time) / rate)


def infer_velocity(rate, gap, first, second):
    if gap <= 0 or rate < 0:
        raise ValueError('Use nonnegative rate and strictly positive time gap.')
    return ((second - first) / gap if rate == 0 else
            rate * (second - first * cosh(rate * gap)) / sinh(rate * gap))


def velocity_bound(rate, gap, error):
    return 2 * error / gap if rate == 0 else (
        rate * error * (1 + cosh(rate * gap)) / sinh(rate * gap))


def predict_position(rate, gap, first, second, time):
    if rate == 0:
        return (1 - time / gap) * first + time / gap * second
    return (sinh(rate * (gap - time)) * first + sinh(rate * time) * second) / sinh(rate * gap)


def position_bound(rate, gap, time, error):
    if rate == 0:
        return error * (abs(1 - time / gap) + abs(time / gap))
    return error * (abs(sinh(rate * (gap - time))) + abs(sinh(rate * time))) / sinh(rate * gap)


def observation_checks():
    count = 0
    for rate, gap, x0, v0 in product([0, .5, 1, 2], [.1, .5, 1, 2], [-1, 0, 1], [-1, .5, 1]):
        last = position(rate, x0, v0, gap)
        assert abs(infer_velocity(rate, gap, x0, last) - v0) < 2e-12
        error = .001
        for da, db in product([-error, error], repeat=2):
            estimate = infer_velocity(rate, gap, x0 + da, last + db)
            assert abs(estimate - v0) <= velocity_bound(rate, gap, error) + 2e-12
            for time in [gap / 2, gap, gap + 1]:
                predicted = predict_position(rate, gap, x0 + da, last + db, time)
                actual = position(rate, x0, v0, time)
                bound = position_bound(rate, gap, time, error)
                assert abs(predicted - actual) <= bound + 2e-11
                if time <= gap:
                    assert bound <= error + 1e-15
        count += 1
    # Opposite signs attain the velocity bound; each bound is not merely a loose
    # consequence of the same numerical inverse operation.
    for gap in [.1, 1, 2]:
        v = infer_velocity(1, gap, -.001, .001)
        assert abs(v - velocity_bound(1, gap, .001)) < 1e-15
        print('Gap, velocity bound, one-second extrapolation bound:', gap,
              velocity_bound(1, gap, .001), position_bound(1, gap, gap + 1, .001))
    print(f'{count} observation states: exact recovery and four noise corners passed')


def regularized_acceleration(x, epsilon, coefficient, family):
    root = sqrt(max(x, 0) + epsilon * epsilon)
    return 12 * coefficient * (root - epsilon if family == 'A' else root)


def release_clock(x, epsilon, coefficient=1.0, steps=4096):
    """Numerical time integral, after x=r^4 removes the endpoint singularity.

    The algebraic factorization avoids subtracting nearly equal cubic roots.
    Simpson comparison is a numerical check, not a certified quadrature bound.
    """
    if x == 0:
        return 0.0
    if epsilon == 0:
        return x ** .25 / sqrt(coefficient)
    upper = x ** .25

    def integrand(r):
        a = sqrt(r ** 4 + epsilon * epsilon)
        return r * sqrt((a + epsilon) / (a * a + a * epsilon + epsilon * epsilon))

    step = upper / steps
    return step * (integrand(0) + integrand(upper) + sum(
        (4 if j % 2 else 2) * integrand(j * step) for j in range(1, steps))) / (3 * sqrt(coefficient))


def regularization_checks():
    count = 0
    for coefficient, epsilon in product([.5, 1, 2], [.1, .01, .001, .0001]):
        assert regularized_acceleration(0, epsilon, coefficient, 'A') == 0
        for x in [-1, 0, 1e-8, .001, 1, 16]:
            original = 12 * coefficient * sqrt(max(x, 0))
            for family in ['A', 'B']:
                assert abs(regularized_acceleration(x, epsilon, coefficient, family) - original) <= 12 * coefficient * epsilon + 2e-13
        for x in [.01, .1, 1, 16]:
            t0 = release_clock(x, 0, coefficient)
            t1 = release_clock(x, epsilon, coefficient)
            t2 = release_clock(x, epsilon, coefficient, 8192)
            bound = 1.25 * sqrt(epsilon / coefficient)
            assert abs(t1 - t2) < 2e-7
            assert -2e-12 <= t0 - t2 <= bound + 2e-12
            count += 1
    print(f'{count} regularized-clock comparisons passed; force errors and rest family checked')


def independent_ode():
    from scipy.integrate import solve_ivp
    count = 0
    for coefficient, epsilon in product([.5, 1, 2], [.1, .01, .001, .0001]):
        alpha = 1.25 * sqrt(epsilon / coefficient)
        for family in ['A', 'B']:
            solution = solve_ivp(lambda t, z: [z[1], regularized_acceleration(z[0], epsilon, coefficient, family)],
                                 [0, 1], [0, 0], method='DOP853', rtol=2e-12,
                                 atol=2e-14 * epsilon ** 2, dense_output=True)
            assert solution.success
            if family == 'A':
                assert all(x == 0 for row in solution.y for x in row)
            else:
                for j in range(21):
                    time = j / 20
                    x, velocity = map(float, solution.sol(time))
                    assert coefficient ** 2 * time ** 4 - 2e-10 <= x <= coefficient ** 2 * (time + alpha) ** 4 + 2e-10
                    root = sqrt(max(x, 0) + epsilon ** 2)
                    potential_difference = x * (root * root + root * epsilon + epsilon ** 2) / (root + epsilon)
                    assert abs(velocity ** 2 - 16 * coefficient * potential_difference) < 2e-9 * max(1, velocity ** 2)
                    position_error_bound = coefficient ** 2 * ((1 + alpha) ** 4 - 1)
                    acceleration_error_bound = 12 * coefficient * (sqrt(position_error_bound) + epsilon)
                    acceleration = regularized_acceleration(x, epsilon, coefficient, 'B')
                    assert abs(acceleration - 12 * coefficient ** 2 * time ** 2) <= acceleration_error_bound + 2e-10
                    assert abs(velocity - 4 * coefficient ** 2 * time ** 3) <= time * acceleration_error_bound + 2e-10
                for time in [.25, .5, 1]:
                    x = float(solution.sol(time)[0])
                    assert abs(release_clock(x, epsilon, coefficient) - time) < 3e-7
                if coefficient == 1:
                    print('epsilon, B position at t=1, analytic upper:', epsilon,
                          solution.y[0, -1], (1 + alpha) ** 4)
            count += 1
    print(f'Independent DOP853: {count} regularized trajectories; 252 B states and 36 clock queries passed')


def figures():
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 12, 'svg.fonttype': 'path',
                         'svg.hashsalt': 'initial-state-research'})
    root = Path('docs/online-learning/physics-math-for-ai/assets')
    root.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        p = root / (name + '.svg')
        fig.savefig(p, metadata={'Date': None}, facecolor='white')
        p.write_text('\n'.join(s.rstrip() for s in p.read_text().splitlines()) + '\n')
        fig.savefig(Path('.runtime/learning') / (name + '.png'), dpi=120, facecolor='white')
        plt.close(fig)

    gap = np.linspace(.05, 2, 201)
    fig, axes = plt.subplots(2, 1, figsize=(5.2, 7), layout='constrained')
    axes[0].semilogy(gap, [velocity_bound(1, d, .001) for d in gap], color='#2563eb')
    axes[0].set(ylabel='Velocity error bound (m/s)', title='Two position observations\nPosition error = 0.001 m')
    axes[1].semilogy(gap, [position_bound(1, d, d + 1, .001) for d in gap], color='#ea580c')
    axes[1].set(xlabel='Observation gap (s)', ylabel='Position error bound (m)',
                title='Extrapolate one second\nbeyond the last observation')
    for ax in axes:
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'initial-observation-conditioning')
    time = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(5.2, 4.8), layout='constrained')
    for epsilon, color in [(.01, '#2563eb'), (.0001, '#ea580c')]:
        alpha = 1.25 * sqrt(epsilon)
        ax.fill_between(time, time ** 4, (time + alpha) ** 4, alpha=.2,
                        color=color, label=f'B enclosure, eps={epsilon}')
    ax.plot(time, time ** 4, color='#2563eb', label='Immediate original solution')
    ax.plot(time, np.zeros_like(time), color='#64748b', ls='--', label='A: stationary for every eps')
    ax.set(xlabel='Time (s)', ylabel='Position (m)', title='Two Lipschitz regularizations\nDifferent selected limits')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=.18)
    ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'initial-regularization-selection')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--independent-ode', action='store_true')
    parser.add_argument('--figures', action='store_true')
    args = parser.parse_args()
    observation_checks()
    regularization_checks()
    if args.independent_ode:
        independent_ode()
    if args.figures:
        figures()
