"""Independent pressure/force integral oracle for isothermal ideal-gas states.

Fraction preserves supplied binary64 inputs. mpmath integrates the equation of
state rather than copying the production log/remainder-gain implementation.
Calculations use 160/1000 digits; JSON stores 80 significant digits explicitly.
"""
import argparse
import json
import math
import random
from fractions import Fraction as F
from pathlib import Path
import mpmath as mp

parser = argparse.ArgumentParser()
parser.add_argument('--output', default='scripts/fixtures/gas-free-energy-independent.json')
out = Path(parser.parse_args().output)
if out.exists():
    raise SystemExit('Refusing to overwrite an existing reference')
R = F(8.31446261815324)
rng = random.Random(26110641)
base = dict(moles=1., temperature=256., externalPressure=float(R * 8192), volume=.03125,
            referenceVolume=.03125, referenceHelmholtz=1000.)


def number(v):
    return mp.mpf(v.numerator) / v.denominator if isinstance(v, F) else mp.mpf(v)


def observation(v):
    v = number(v)
    return dict(value=mp.nstr(v, 80), sign=int(mp.sign(v)),
                logMagnitude=None if v == 0 else mp.nstr(mp.log(abs(v)), 80))


def case(p, digits, derivatives=False):
    with mp.workdps(digits):
        n, T, P, V, Vr, anchor = [F(p[k]) for k in ['moles', 'temperature', 'externalPressure', 'volume', 'referenceVolume', 'referenceHelmholtz']]
        A = n * R * T
        star, balance = A / P, P * V - A
        a, pressure, v, vr, eq = map(number, [A, P, V, Vr, star])

        def pressure_work(left, right):
            return mp.mpf(0) if left == right else mp.quad(lambda z: a / z, [number(left), number(right)])

        work_from_reference = pressure_work(Vr, V)
        work_equilibrium_reference = pressure_work(Vr, star)
        log_reference = work_from_reference / a
        work_from_equilibrium = pressure_work(star, V)
        displacement = number(V - star)
        # Integrate (pext-pgas)dV on a normalized interval. This is positive
        # for either sign of displacement, and retains arbitrarily small gaps.
        gap = mp.mpf(0) if balance == 0 else pressure * displacement**2 * mp.quad(
            lambda u: u / (eq + u * displacement), [0, 1])
        free = number(anchor) - work_from_reference
        gas_gibbs = free + a
        phi = free + pressure * v
        gibbs_eq = number(anchor) - work_equilibrium_reference + a
        # Recognize only algebraic zeros, never round a numerical residual away.
        if V == Vr:
            free = number(anchor)
            gas_gibbs = number(anchor + A)
            phi = number(anchor + P * V)
        if star == Vr:
            gibbs_eq = number(anchor + A)
        expected = dict(
            thermalEnergyScale=A, equilibriumVolume=star, volumeRatioToEquilibrium=P * V / A,
            relativeVolumeDisplacement=balance / A, volumeDisplacement=V - star,
            pressureVolumeMinusThermalEnergy=balance, gasPressure=A / V,
            externalMinusGasPressure=balance / V, constrainedPotentialCurvature=A / V**2,
            logVolumeRatioFromReference=log_reference, logVolumeRatioToEquilibrium=work_from_equilibrium / a,
            helmholtzChangeFromReference=-work_from_reference,
            entropyChangeFromReference=work_from_reference / number(T), internalEnergyChangeFromReference=F(0),
            helmholtzFreeEnergy=free, gasGibbsFreeEnergy=gas_gibbs, constrainedPotential=phi,
            gibbsAtExternalPressureEquilibrium=gibbs_eq, constrainedMinusGasGibbs=balance,
            helmholtzChangeFromEquilibrium=-work_from_equilibrium,
            externalPressureVolumePotentialChangeFromEquilibrium=balance,
            availability=gap, availabilityPerThermalEnergy=gap / a,
        )
        budget = dict(energyChangeGas=F(0), heatIntoGas=-balance, workByGas=-balance,
                      entropyChangeGas=-work_from_equilibrium / number(T),
                      entropyChangeThermalReservoir=balance / T, entropyProduction=gap / number(T),
                      maximumReversibleWorkByGas=-work_from_equilibrium, workDeficitFromReversible=gap)
        if derivatives:
            phi_function = lambda w: number(anchor) - a * mp.log(w / vr) + pressure * w
            tolerance = mp.power(10, -digits // 2)
            assert abs(mp.diff(phi_function, v) - number(balance / V)) < tolerance * max(1, pressure)
            assert abs(mp.diff(phi_function, v, 2) - number(A / V**2)) < tolerance * max(1, number(A / V**2))
        assert (gap == 0) == (balance == 0)
        assert gap >= 0
        return dict(parameters=p, digits=digits, derivativeCheck=derivatives,
                    regime='equilibrium' if balance == 0 else 'below-equilibrium-volume' if balance < 0 else 'above-equilibrium-volume',
                    equilibriumConditionSatisfied=balance == 0,
                    expected={k: observation(value) for k, value in expected.items()},
                    conditionalEquilibrationBudget={k: observation(value) for k, value in budget.items()})


ordinary_inputs = [base, {**base, 'volume': .015625}, {**base, 'volume': .0625},
                   {**base, 'referenceVolume': .0625},
                   dict(moles=1., temperature=300., externalPressure=100000., volume=.05, referenceVolume=.025, referenceHelmholtz=0.),
                   dict(moles=.25, temperature=250., externalPressure=200000., volume=.001, referenceVolume=1., referenceHelmholtz=-10000.),
                   dict(moles=2., temperature=800., externalPressure=25000., volume=1., referenceVolume=.001, referenceHelmholtz=10000.),
                   {**base, 'referenceHelmholtz': -float(R * 256)}]
for _ in range(16):
    ordinary_inputs.append(dict(moles=rng.uniform(.25, 2), temperature=rng.uniform(250, 800),
                                externalPressure=rng.uniform(25000, 200000), volume=rng.uniform(.001, 1),
                                referenceVolume=rng.uniform(.001, 1), referenceHelmholtz=rng.uniform(-10000, 10000)))
ordinary = [case(p, 160, i < 4) for i, p in enumerate(ordinary_inputs)]
limit_inputs = []
for count in [1, 2, 16, 64]:
    for direction in [-math.inf, math.inf]:
        volume = base['volume']
        for _ in range(count):
            volume = math.nextafter(volume, direction)
        limit_inputs.append({**base, 'volume': volume})
for n, T, P in [(1., 300., 100000.), (.3, 777.7, 123456.7), (1.7, 270.1, 31001.3), (.25, 800., 199999.9)]:
    star = F(n) * R * F(T) / F(P)
    limit_inputs.append({**base, 'moles': n, 'temperature': T, 'externalPressure': P, 'volume': float(star)})
for ratio in [.875, 1.125]:
    boundary = base['volume'] * ratio
    for volume in [math.nextafter(boundary, -math.inf), boundary, math.nextafter(boundary, math.inf)]:
        limit_inputs.append({**base, 'volume': volume})
# ln(binary64 e) is not mathematically one. The production Math.log rounds it
# to one, allowing a computed-zero anchored F or G without a structural zero.
limit_inputs += [{**base, 'volume': math.e / 32, 'referenceHelmholtz': float(R * 256)},
                 {**base, 'volume': math.e / 32, 'referenceHelmholtz': 0.}]
limits = [case(p, 1000) for p in limit_inputs]
assert len(ordinary) == 24 and len(limits) == 20
out.write_text(json.dumps(dict(
    method='Independent pressure-work and positive mechanical-force quadrature, exact Fraction input balances; four numerical derivative checks.',
    serializedSignificantDigits=80, ordinary=ordinary, limits=limits), indent=2) + '\n')
print(f'Wrote {len(ordinary)} ordinary (160 digits) and {len(limits)} limit (1000 digits) references to {out}')
