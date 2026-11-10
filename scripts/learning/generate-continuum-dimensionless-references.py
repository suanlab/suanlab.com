"""Independent high-precision direct BVP formulas, derivatives and physical-coordinate quadrature.

No production imports. Binary64 input numbers are converted exactly to mpmath.
"""
import argparse
import json
import random
from pathlib import Path
import mpmath as mp

parser = argparse.ArgumentParser()
parser.add_argument('--output', default='scripts/fixtures/continuum-dimensionless-independent.json')
target = Path(parser.parse_args().output)
if target.exists():
    raise SystemExit('Refusing to overwrite an existing reference')
mp.mp.dps = 160
M = mp.mpf
rng = random.Random(261106)


def observation(v):
    return dict(value=str(v), sign=int(mp.sign(v)), logMagnitude=None if v == 0 else str(mp.log(abs(v))))


def pe_case(p, xi, quadrature=False):
    D, L, A, U, left, right = [M(p[key]) for key in ['diffusivity', 'length', 'area', 'velocity', 'leftConcentration', 'rightConcentration']]
    x = M(xi) * L
    P = U * L / D
    theta = (lambda s: s / L) if U == 0 else (lambda s: mp.expm1(U * s / D) / mp.expm1(P))
    left_weight = (lambda s: 1-s/L) if U == 0 else (lambda s: (mp.exp(P)-mp.exp(U*s/D))/mp.expm1(P))
    field = lambda s: left * left_weight(s) + right * theta(s)
    c = field(x)
    gradient = mp.diff(field, x)
    # Evaluate at the inlet for positive flow and the outlet for negative flow.
    # This retains the tiny opposite-boundary flux without subtracting local large contributions.
    endpoint = M(0) if U >= 0 else L
    flux = U * field(endpoint) - D * mp.diff(field, endpoint)
    expected = dict(peclet=P, basisShape=theta(x), leftWeight=left_weight(x), stableShapeCorrection=theta(x)-M(xi),
                    concentration=c, gradient=gradient, curvature=mp.diff(field, x, 2),
                    advectiveFlux=U*c, diffusiveFlux=-D*gradient, totalFlux=flux, totalTransport=A*flux,
                    diffusionTime=L*L/D)
    if U != 0:
        expected.update(advectionTime=L/abs(U), boundaryLayerLength=D/abs(U))
    out = dict(parameters=p, xi=xi, expected={key: observation(v) for key, v in expected.items()})
    if quadrature:
        # Differentiate the actual field, then integrate its transport along the physical interval.
        integral = mp.quad(lambda s: U*field(s)-D*mp.diff(field, s), [0, L/2, L])
        assert mp.almosteq(integral, L*flux)
        out['physicalFluxIntegral'] = observation(integral)
    return out


def da_case(p, xi, quadrature=False):
    D, L, A, k, c0 = [M(p[key]) for key in ['diffusivity', 'length', 'area', 'reactionRate', 'surfaceConcentration']]
    x = M(xi)*L
    Da = k*L*L/D
    phi = mp.sqrt(Da)
    shape = (lambda s: M(1)) if k == 0 else (lambda s: mp.cosh(mp.sqrt(k/D)*(L-s))/mp.cosh(phi))
    field = lambda s: c0*shape(s)
    effectiveness = M(1) if k == 0 else mp.tanh(phi)/phi
    mass = A*c0*L*effectiveness
    partial_mass = A*c0*x if k == 0 else A*c0*(mp.sinh(phi)-mp.sinh(mp.sqrt(k/D)*(L-x)))/(mp.sqrt(k/D)*mp.cosh(phi))
    gradient = mp.diff(field, x)
    expected = dict(damkohler=Da, thiele=phi, basisShape=shape(x), stableDepletion=1-shape(x),
                    concentration=field(x), gradient=gradient, curvature=mp.diff(field, x, 2),
                    diffusiveFlux=-D*gradient, localConsumption=k*field(x), storedMass=mass,
                    totalConsumption=k*mass, inletFlux=-D*mp.diff(field, M(0)),
                    partialStoredMass=partial_mass, partialConsumption=k*partial_mass,
                    analyticEffectiveness=effectiveness, stableEffectivenessDeficit=1-effectiveness,
                    diffusionTime=L*L/D)
    if k > 0:
        expected.update(reactionTime=1/k, penetrationLength=mp.sqrt(D/k))
    out = dict(parameters=p, xi=xi, expected={key: observation(v) for key, v in expected.items()})
    if quadrature:
        physical_mass = A*mp.quad(field, [0, L/2, L])
        physical_consumption = A*mp.quad(lambda s: k*field(s), [0, L/2, L])
        assert mp.almosteq(physical_mass, mass)
        assert mp.almosteq(physical_consumption, -A*D*mp.diff(field, M(0)))
        assert mp.almosteq(partial_mass, A*mp.quad(field, [0, x]))
        out.update(physicalMassIntegral=observation(physical_mass), physicalConsumptionIntegral=observation(physical_consumption))
    return out


peclet = []
damkohler = []
for _ in range(20):
    common = dict(diffusivity=rng.uniform(.1, 2), length=rng.uniform(.3, 3), area=rng.uniform(.3, 3))
    peclet.append(pe_case(dict(**common, velocity=rng.uniform(-1.5, 1.5), leftConcentration=rng.uniform(.1, 3), rightConcentration=rng.uniform(.1, 3)), rng.uniform(.05, .95), True))
    damkohler.append(da_case(dict(**common, reactionRate=rng.uniform(.1, 5), surfaceConcentration=rng.uniform(.1, 3)), rng.uniform(.05, .95), True))

pe_limits = []
da_limits = []
with mp.workdps(1200):
    base = dict(diffusivity=.1, length=2., area=1.5, velocity=.5, leftConcentration=1., rightConcentration=.2)
    for edits, xi in [({}, .5), ({'velocity': 0.}, .5), ({'velocity': 1e-20}, .4),
                      ({'velocity': 5e-324, 'diffusivity': 4., 'length': .25}, 5e-324),
                      ({'velocity': -.5}, .2), ({'velocity': 2., 'length': 4., 'diffusivity': 1e-4, 'leftConcentration': 0., 'rightConcentration': 1.}, .5),
                      ({'velocity': -2., 'length': 4., 'diffusivity': 1e-4, 'leftConcentration': 1., 'rightConcentration': 0.}, .5),
                      ({'leftConcentration': 2., 'rightConcentration': 2.}, .7),
                      ({'leftConcentration': 0., 'rightConcentration': 0.}, .2),
                      ({'velocity': 0., 'leftConcentration': 0., 'rightConcentration': 5e-324}, .5),
                      ({'leftConcentration': 0., 'rightConcentration': 1.}, 0.),
                      ({'leftConcentration': 0., 'rightConcentration': 1.}, 1.)]:
        pe_limits.append(pe_case(dict(base, **edits), xi))
    base = dict(diffusivity=.25, length=1., area=1.5, reactionRate=1., surfaceConcentration=2.)
    for edits, xi in [({}, 1.), ({'reactionRate': 0.}, .5), ({'reactionRate': 1e-20}, .5),
                      ({'reactionRate': 5e-324, 'diffusivity': 4., 'length': .25}, 5e-324),
                      ({'reactionRate': 128., 'length': 4., 'diffusivity': 1e-4}, 1.),
                      ({'surfaceConcentration': 0.}, .7), ({'surfaceConcentration': 5e-324, 'area': 4., 'length': 4.}, .5),
                      ({'reactionRate': 5e-324, 'surfaceConcentration': 5e-324}, .5),
                      ({'reactionRate': 1e-200}, 1.), ({'reactionRate': 1e-6}, .4),
                      ({'reactionRate': 128., 'length': 4., 'diffusivity': 1e-4}, 0.),
                      ({'reactionRate': 1e-20}, 0.)]:
        da_limits.append(da_case(dict(base, **edits), xi))

target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(dict(method='direct exponential/cosh BVP; mpmath derivatives; independent physical-coordinate quadrature', ordinaryDigits=160, limitDigits=1200,
                                  pecletCases=peclet, damkohlerCases=damkohler, pecletLimits=pe_limits, damkohlerLimits=da_limits), indent=2)+'\n')
print(f'Wrote {len(peclet)+len(damkohler)+len(pe_limits)+len(da_limits)} independent references to {target}')
