"""Independent soft-endpoint action and full oscillator variation checks.

Only the Python standard library is required. The terminal penalty and
additional Fourier modes are supplementary examples, not new native controls.
"""
from fractions import Fraction as Q
from itertools import product
from math import sin, cos, pi


def polynomial_action(mass, duration, start, end, bow, odd, gamma):
    # Exact five-node Boole quadrature for the degree-five gauged integrand.
    samples = []
    for j in range(5):
        s = Q(j, 4)
        position = start+(end-start)*s+bow*s*(1-s)+odd*s*(1-s)*(2*s-1)
        velocity = ((end-start)+bow*(1-2*s)+odd*(-6*s*s+6*s-1))/duration
        samples.append(mass*velocity*velocity/2+gamma*position*velocity)
    return duration*sum(w*v for w, v in zip([7, 32, 12, 32, 7], samples))/90


def endpoint_objective(mass, duration, start, end, target, weight, gamma):
    return (mass*(end-start)**2/(2*duration)+gamma*(end*end-start*start)/2
            +weight*(end-target)**2/2)


def endpoint_checks():
    count = 0
    for m, t, weight, start, target, gamma in product(
            [1, 2], [1, 2], [1, 2], [-1, 0], [0, 1], [-4, -2, 0, 2]):
        m, t, weight, start, target, gamma = map(Q, [m, t, weight, start, target, gamma])
        hessian = m/t+gamma+weight
        drive = m*start/t+weight*target
        for end in map(Q, [-1, 0, 2]):
            step = Q(1, 7)
            derivative = (endpoint_objective(m, t, start, end+step, target, weight, gamma)
                          -endpoint_objective(m, t, start, end-step, target, weight, gamma))/(2*step)
            natural = m*(end-start)/t+gamma*end+weight*(end-target)
            assert derivative == natural == hessian*end-drive
            for bow, odd in [(Q(0), Q(0)), (Q(1), Q(0)), (Q(1), Q(-1))]:
                gauged = polynomial_action(m, t, start, end, bow, odd, gamma)
                base = polynomial_action(m, t, start, end, bow, odd, Q(0))
                terminal = weight*(end-target)**2/2
                shift = gamma*(end*end-start*start)/2
                assert gauged-base == shift
                assert gauged+terminal-shift == base+terminal
                reduced = endpoint_objective(m, t, start, end, target, weight, gamma)
                assert gauged+terminal-reduced == m*(bow*bow/3+odd*odd/5)/(2*t)
                count += 1
        if hessian:
            stationary = drive/hessian
            assert m*(stationary-start)/t+gamma*stationary+weight*(stationary-target) == 0
            for perturbation in [Q(-1), Q(1, 2), Q(1)]:
                difference = (endpoint_objective(m, t, start, stationary+perturbation, target, weight, gamma)
                              -endpoint_objective(m, t, start, stationary, target, weight, gamma))
                assert difference == hessian*perturbation**2/2
    # The zero-curvature distinction: a nonzero drive is unbounded; zero drive is flat.
    assert endpoint_objective(Q(1), Q(1), Q(0), Q(4), Q(1), Q(1), Q(-2)) == Q(-7, 2)
    for end in map(Q, [-2, 0, 3]):
        assert endpoint_objective(Q(1), Q(1), Q(0), end, Q(0), Q(1), Q(-2)) == 0
    for gamma in map(Q, [-3, -2, -1, 0, 1]):
        hessian = 2+gamma
        minimum = Q(1)/hessian if hessian > 0 else None
        print('gamma, endpoint curvature, uncompensated minimum, corrected minimum:',
              gamma, hessian, minimum, Q(1, 2))
    print(f'{count} exact free-endpoint paths passed; gauge/terminal cancellation and natural derivatives checked')


def morse_signature(ratio):
    ratio = Q(ratio)
    if ratio < 0:
        raise ValueError('Nonnegative frequency-time ratio required.')
    ceiling = (ratio.numerator+ratio.denominator-1)//ratio.denominator
    return max(0, ceiling-1), int(ratio > 0 and ratio.denominator == 1)


def morse_checks():
    horizons = [Q(f*n, 40) for f in range(5) for n in range(5, 26)]
    horizons += [Q(n)+epsilon for n in range(1, 5) for epsilon in [Q(-1, 1000), Q(0), Q(1, 1000)]]
    for ratio in horizons:
        negative, nullity = morse_signature(ratio)
        coefficients = [Q(n*n)-ratio*ratio for n in range(1, 8)]
        assert sum(v < 0 for v in coefficients) == negative
        assert sum(v == 0 for v in coefficients) == nullity
        assert coefficients[-1] > 0
    count = 0
    coefficients = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [1, -1, .5, 0, 1/3]]
    for ratio, mass, duration, amplitudes in product(
            [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4], [.5, 1, 2], [1, 2], coefficients):
        omega = ratio*pi/duration
        numeric = 0.
        # Midpoint integration independently evaluates physical kinetic-minus-potential terms.
        for j in range(256):
            s = (j+.5)/256
            position = sum(c*sin(n*pi*s) for n, c in enumerate(amplitudes, 1))
            velocity = sum(c*n*pi*cos(n*pi*s)/duration for n, c in enumerate(amplitudes, 1))
            numeric += mass*(velocity*velocity-omega*omega*position*position)*duration/256
        exact = mass*pi*pi/(2*duration)*sum((n*n-ratio*ratio)*c*c for n, c in enumerate(amplitudes, 1))
        assert abs(numeric-exact) < 2e-11*max(1, abs(exact))
        count += 1
    for ratio in map(Q, ['1/2', '1', '3/2', '2', '5/2', '3']):
        print('ratio, negative directions, zero directions:', ratio, *morse_signature(ratio))
    print(f'{len(horizons)} exact Morse signatures and {count} independent quadratic-form integrals passed')


def boundary_velocity(ratio, duration, start, end):
    ratio = Q(ratio)
    if duration <= 0 or ratio < 0:
        raise ValueError('Positive duration and nonnegative ratio required.')
    if ratio == 0:
        return {'status': 'unique', 'velocity': (end-start)/duration}
    if ratio.denominator == 1:
        return {'status': 'family' if end == (-1)**ratio.numerator*start else 'no-solution'}
    omega = float(ratio)*pi/duration
    velocity = omega*(end-start*cos(float(ratio)*pi))/sin(float(ratio)*pi)
    return {'status': 'unique', 'velocity': velocity}


def boundary_checks():
    statuses = {'family': 0, 'no-solution': 0}
    for n, start, end in product(range(1, 5), [-1, 0, 1], [-1, 0, 1]):
        result = boundary_velocity(Q(n), n, start, end)
        statuses[result['status']] += 1
        if result['status'] == 'family':
            # Exact critical-time compatibility avoids dividing by roundoff sin(n*pi).
            assert (-1)**n*start == end
    assert statuses == {'family': 12, 'no-solution': 24}
    count = 0
    for ratio, start, end, error in product(
            map(Q, ['1/2', '99/100', '101/100', '199/100', '201/100']), [0, 1], [-1, 1], [-.001, .001]):
        duration = float(ratio)  # omega=pi rad/s, duration=ratio seconds.
        velocity = boundary_velocity(ratio, duration, start, end)['velocity']
        noisy = boundary_velocity(ratio, duration, start, end+error)['velocity']
        gain = pi/sin(float(ratio)*pi)
        assert abs((noisy-velocity)-gain*error) < 3e-11
        recovered = start*cos(float(ratio)*pi)+velocity/pi*sin(float(ratio)*pi)
        assert abs(recovered-end) < 2e-12
        count += 1
    assert boundary_velocity(Q(0), 2, 1, 3)['velocity'] == 1
    print(f'36 critical endpoint cases and {count} near-critical endpoint perturbations passed')


if __name__ == '__main__':
    endpoint_checks()
    morse_checks()
    boundary_checks()
