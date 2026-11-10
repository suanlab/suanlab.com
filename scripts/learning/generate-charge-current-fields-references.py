"""Independent NumPy/SciPy electrostatic and charge-transport references.

No TypeScript product imports. k is factored out. Potential finite differences,
space/time quadratures and independent derivative stencils are distinct checks.
"""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

rng = np.random.default_rng(91302)
fields, currents, intervals = [], [], []
for i in range(48):
    positions = rng.uniform(-1, 1, (3 + i % 3, 2))
    charges = rng.uniform(-2, 2, len(positions))
    query = rng.uniform(1.5, 3, 2)
    displacement = query - positions
    distance = np.linalg.norm(displacement, axis=1)
    field = np.sum(charges[:, None] * displacement / distance[:, None]**3, axis=0)

    def potential(x):
        return float(np.sum(charges / np.linalg.norm(x - positions, axis=1)))

    h = 1e-4
    grad = []
    for j in range(2):
        e = np.eye(2)[j] * h
        grad.append((-potential(query + 2*e) + 8*potential(query + e)
                     - 8*potential(query - e) + potential(query - 2*e)) / (12*h))
    assert np.max(np.abs(field + grad)) < 2e-10
    fields.append(dict(sources=[dict(position=p.tolist(), charge=float(q)) for p, q in zip(positions, charges)],
                       observation=query.tolist(), fieldOverK=field.tolist(),
                       potentialOverK=potential(query), negativePotentialGradient=(-np.array(grad)).tolist()))

for i in range(60):
    s = dict(totalCharge=[-2., 0., 1.][i % 3], sigma=[.1, .3, 1., 2.][i % 4],
             initialCenter=[-1., .2, 1.][(i // 3) % 3], velocity=[-2., -.7, 0., .4, 2.][i % 5],
             area=[.1, 1., 4.][(i // 4) % 3])
    left, right, time, start = -.5, .8, float(np.linspace(-2, 2, 7)[i % 7]), -.7

    def rho(x, t):
        z = (x - s['initialCenter'] - s['velocity'] * t) / s['sigma']
        return s['totalCharge'] / (s['area'] * s['sigma'] * np.sqrt(2*np.pi)) * np.exp(-z*z/2)

    def charge(t):
        center = s['initialCenter'] + s['velocity'] * t
        z0, z1 = (left-center)/s['sigma'], (right-center)/s['sigma']
        # Opposite-tail CDF avoids subtracting two values near one.
        mass = ndtr(-z0)-ndtr(-z1) if z0 > 0 else ndtr(z1)-ndtr(z0)
        return s['totalCharge'] * mass

    center = s['initialCenter'] + s['velocity'] * time
    amount_quad = s['area'] * quad(lambda x: rho(x, time), left, right,
                                  points=[center] if left < center < right else None,
                                  epsabs=1e-13, epsrel=2e-12)[0]
    inflow = lambda t: s['area'] * s['velocity'] * (rho(left, t) - rho(right, t))
    temporal = quad(inflow, start, time, epsabs=1e-12, epsrel=2e-12)[0]
    h = 1e-4
    dt = (-charge(time+2*h) + 8*charge(time+h) - 8*charge(time-h) + charge(time-2*h))/(12*h)
    observations = []
    for x in [-1., .1, 2.]:
        rho_dt = (-rho(x, time+2*h) + 8*rho(x, time+h) - 8*rho(x, time-h) + rho(x, time-2*h))/(12*h)
        rho_dx = (-rho(x+2*h, time) + 8*rho(x+h, time) - 8*rho(x-h, time) + rho(x-2*h, time))/(12*h)
        observations.append(dict(x=x, density=rho(x, time), currentDensity=s['velocity']*rho(x, time),
                                 independentTimeDerivative=rho_dt, independentCurrentDivergence=s['velocity']*rho_dx))
    assert abs(charge(time)-amount_quad) < 3e-13
    assert abs((charge(time)-charge(start))-temporal) < 3e-12
    assert abs(dt-inflow(time)) < 5e-8
    currents.append(dict(system=s, left=left, right=right, time=time, startTime=start,
                         chargeInside=charge(time), independentSpaceIntegral=amount_quad,
                         netInflow=inflow(time), independentTimeDerivative=dt,
                         chargeChange=charge(time)-charge(start), independentFlowIntegral=temporal,
                         observations=observations))

# Narrow positive tails where naive Phi(b)-Phi(a) becomes exactly zero in doubles.
for left, right in [(8., 8.+1e-10), (9., 9.25), (12., 12.5), (-12., -11.9),
                    (0., 1e-13), (-16., 16.), (8., np.nextafter(8., np.inf))]:
    mass = quad(lambda z: np.exp(-z*z/2)/np.sqrt(2*np.pi), left, right,
                epsabs=1e-300, epsrel=2e-13, points=[0.] if left < 0 < right else None)[0]
    intervals.append(dict(left=left, right=right, fraction=mass))

result = dict(provenance='NumPy direct vector sum; SciPy ndtr/quad and independent five-point derivatives; no product imports',
              fields=fields, currents=currents, tailIntervals=intervals)
path = Path('scripts/fixtures/charge-current-fields-reference.json')
path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps(dict(fields=len(fields), transports=len(currents), localObservations=3*len(currents), tailIntervals=len(intervals))))
