"""Independent references: SciPy DOP853 augmented work/heat integration.

No TypeScript implementation is imported. Frequencies include resonance, DC,
near resonance, damping boundaries and maximum supported time/frequency.
Run from repository root: python3 scripts/learning/generate-waves-oscillations-references.py
"""
import json
import math
import random
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp, quad
from scipy.linalg import expm

rng = random.Random(20261105)
cases = []
special = [
    (1, 1, 0, 1, 1, 0, 0, math.pi / 2),
    (1, 1, 0, 1, 1 + 1e-10, 0, 0, 32),
    (1, 1, 1, 1, 1, 0, 1, 2 * math.pi),
    (1, 1, 2, 1, 1, 1, -1, 8),
    (1, 1, 4, 1, 0, 0, 0, 8),
    (1, 2, 2, 1, 1, 0, 0, 8),
    (.25, 4, 8, 4, 8, 4, -4, 32),
    (4, .25, 0, 4, 8, -4, 4, 32),
    (1, 1, 0, 0, 1, 1, 0, 8),
    (1, 1, 1, 0, 0, 0, 0, 0),
]
for i in range(48):
    special.append((rng.choice([.25, 1, 2, 4]), rng.choice([.25, 1, 2, 4]),
                    rng.choice([0, .25, 1, 2, 4, 8]), rng.choice([0, 1, 2, 4]),
                    rng.choice([0, .5, 1, 2, 4, 8]), rng.uniform(-4, 4),
                    rng.uniform(-4, 4), rng.uniform(.1, 12)))
for m, k, c, f, w, x, v, duration in special:
    def rhs(t, y):
        force = f * math.cos(w * t)
        return [y[1], (force - c * y[1] - k * y[0]) / m, force * y[1], c * y[1] ** 2]
    times = np.linspace(0, duration, 9)
    if duration == 0:
        states = np.array([[x, v, 0, 0]] * 9)
        refined = states
    else:
        def solve(tol):
            result = solve_ivp(rhs, [0, duration], [x, v, 0, 0], method='DOP853',
                               rtol=tol, atol=tol * .05, t_eval=times)
            assert result.success, result.message
            return result.y.T
        states, refined = solve(2e-11), solve(3e-13)
    matrix = np.array([[0, 1, 0, 0], [-k / m, -c / m, f / m, 0],
                       [0, 0, 0, -w], [0, 0, w, 0]], dtype=float)
    linear = (expm(matrix * duration) @ [x, v, 1, 0])[:2]
    assert np.max(np.abs(linear - refined[-1, :2])) < 2e-9
    cases.append(dict(system=dict(mass=m, stiffness=k, damping=c, forceAmplitude=f, frequency=w),
                      initial=dict(position=x, velocity=v), duration=duration,
                      samples=[dict(time=float(t), position=float(y[0]), velocity=float(y[1])) for t, y in zip(times, refined)],
                      work=float(refined[-1, 2]), heat=float(refined[-1, 3]),
                      refinementDifference=float(np.max(np.abs(states - refined))),
                      independentExpmDifference=float(np.max(np.abs(linear - refined[-1, :2])))))

power_cases = []
for m, k, c, f in [(1, 1, .25, 1), (1, 1, 1, 1), (1, 2, 2, 2), (.25, 4, 8, 4), (4, .25, .25, 1)]:
    natural = math.sqrt(k / m)
    root = math.sqrt(c * c + 4 * m * k)
    for w in [natural, 2 * k / (root + c), (root + c) / (2 * m), natural * .6, natural * 1.7]:
        # Fixtures for the frequency model may exceed the transient frequency UI
        # range; include only values supported by the shared public API.
        if w > 8:
            continue
        z = complex(k - m * w * w, c * w)
        chi = 1 / z
        def velocity(t):
            return (1j * w * f * chi * np.exp(1j * w * t)).real
        period = 2 * math.pi / w
        work = quad(lambda t: f * math.cos(w * t) * velocity(t), 0, period, epsabs=1e-11, epsrel=1e-11)[0]
        heat = quad(lambda t: c * velocity(t) ** 2, 0, period, epsabs=1e-11, epsrel=1e-11)[0]
        power_cases.append(dict(system=dict(mass=m, stiffness=k, damping=c, forceAmplitude=f, frequency=w),
                                meanPower=work / period, meanHeat=heat / period,
                                gain=abs(chi), lag=-math.atan2(chi.imag, chi.real)))

output = dict(provenance=dict(generator=__file__, scipy=scipy.__version__, seed=20261105,
                               method='DOP853 tolerance refinement, SciPy expm cross-check, real cycle quadrature'),
              trajectories=cases, periodic=power_cases)
path = Path('scripts/fixtures/waves-oscillations-independent.json')
path.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
print(f'{path}: {len(cases)} trajectories, {len(power_cases)} periodic integrals')
