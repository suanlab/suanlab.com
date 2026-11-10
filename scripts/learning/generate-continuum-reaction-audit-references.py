"""Independent 1600-digit kinetics and single-round binary64 observations."""
import argparse
import json
from pathlib import Path
import mpmath as mp

mp.mp.dps = 1600
parser = argparse.ArgumentParser()
parser.add_argument('--output', default='scripts/fixtures/continuum-reaction-audit-independent.json')
path = Path(parser.parse_args().output)
if path.exists():
    raise SystemExit('Refusing to overwrite an existing reference')

def observation(q):
    if q == 0:
        return dict(value=0., sign=0)
    # float(mpf) can first round to 53 significant bits, then round again at
    # a subnormal midpoint. Round the integer quantum once, ties to even.
    magnitude = abs(q)
    _, exponent = mp.frexp(magnitude)
    quantum = max(-1074, exponent-53)
    scaled = mp.ldexp(magnitude, -quantum)
    mantissa = int(mp.floor(scaled))
    remainder = scaled-mantissa
    if remainder > mp.mpf('.5') or remainder == mp.mpf('.5') and mantissa % 2:
        mantissa += 1
    rounded = int(mp.sign(q))*float(mantissa)*2.**quantum
    return dict(value=None if rounded == 0 else rounded, sign=int(mp.sign(q)))

minimum = float.fromhex('0x0.0000000000001p-1022')
m = mp.mpf(minimum)
assert observation(m*mp.mpf('1.5')*(1-m))['value'] == minimum
assert observation(m*mp.mpf('2.5')*(1+m))['value'] == 3*minimum
assert observation(m/2)['value'] is None

cases = []
rates = [(minimum,minimum), (minimum,0.), (0.,minimum), (3*minimum,minimum),
         (1e-200,1e-200), (1e-20,0.), (1e-8,1e-8), (2e-8,3e-8)]
settings = [(kf,kb,a) for kf,kb in rates for a in [0., .25, .5, .75, 1.]]
settings += [(2.,1.,1/3), (1.,2.,2/3), (3.,1.,.25)]
for forward, reverse, initial in settings:
    for time in [0., .5, 1., 2., 3., 4., minimum]:
        kf,kb,a0,t = map(mp.mpf,[forward,reverse,initial,time])
        rate=kf+kb
        equilibrium_a=kb/rate
        equilibrium_b=kf/rate
        e=mp.exp(-rate*t)
        mix=-mp.expm1(-rate*t)
        a=a0*e+equilibrium_a*mix
        b=(1-a0)*e+equilibrium_b*mix
        d0=kb*(1-a0)-kf*a0
        derivative=d0*e
        change=d0*mix/rate
        carrier=3/mp.sqrt(2*mp.pi)
        values=dict(fractionA=a, fractionB=b,
            equilibriumA=equilibrium_a,equilibriumB=equilibrium_b,
            fractionRateA=derivative,stableFractionChangeA=change,
            globalMassA=3*a,globalMassB=3*b,
            globalSourceA=3*derivative,globalSourceB=-3*derivative,
            localSourceA=carrier*derivative)
        cases.append(dict(parameters=dict(forwardRate=forward,reverseRate=reverse,
            initialFractionA=initial), time=time,
            expected={key:observation(q) for key,q in values.items()},
            logRateMagnitude=None if derivative == 0 else float(mp.log(abs(derivative)))))
path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(dict(precisionDigits=1600,
    method='Independent exponential kinetics, Gaussian source and integer-quantum ties-to-even projection',
    cases=cases),indent=2,allow_nan=False)+'\n')
print(f'Wrote {len(cases)} independent kinetic cases')
