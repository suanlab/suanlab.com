"""Independent high-precision totals; no product imports or fixture overwrites."""
import argparse
import json
from pathlib import Path
import mpmath as mp

mp.mp.dps = 1600
parser = argparse.ArgumentParser()
parser.add_argument('--output', default='scripts/fixtures/continuum-heat-total-independent.json')
path = Path(parser.parse_args().output)
if path.exists():
    raise SystemExit('Refusing to overwrite an existing reference')

def observation(q):
    rounded = float(q)
    return {'value': None if q != 0 and rounded == 0 else rounded,
            'sign': int(mp.sign(q)), 'decimal': mp.nstr(q, 100)}

cases = []
minimum = float.fromhex('0x0.0000000000001p-1022')
settings = [
    ('under-range-contrast', minimum, 4., 4., 1.),
    ('negative-under-range-contrast', -minimum, 4., 4., 1.),
    ('ordinary-contrast-late-decay', .2, 4., 4., 1.),
    ('rescued-amplitude', .2, 4., 749/256, 1.),
    ('tiny-positive-window', .1, 1., .5, minimum),
    ('tiny-negative-window', -.1, 1., .5, minimum),
    ('exact-zero-width', .1, 1., .5, 0.),
    ('ordinary-positive', .1, 1., .5, .7),
    ('ordinary-negative', -.1, 1., .5, .7),
    ('zero-contrast', 0., 4., 4., 1.),
    ('zero-conductivity', minimum, 0., 4., 1.),
]
for mean in [250., 273.15, 300., 400.]:
    for name, contrast, conductivity, time, half_width in settings:
        p = dict(heatCapacity=.25, conductivity=conductivity,
                 meanTemperature=mean, contrast=contrast,
                 waveNumber=4., phase=0., area=1.)
        C, K, T, a, k, A, t, h = map(mp.mpf,
            [.25, conductivity, mean, contrast, 4., 1., time, half_width])
        amplitude = T*a*mp.exp(-K*k*k*t/C)
        # Analytically integrated physical window [-h,h], independently
        # evaluated at 1600 digits, including the declared binary64 reference.
        mode_heat = C*A*amplitude*2*mp.sin(k*h)/k
        total_heat = C*A*2*h*(T-mp.mpf(273.15)) + mode_heat
        cases.append(dict(name=f'{name}-{mean}', parameters=p, x=0.,
            center=0., halfWidth=half_width, time=time,
            expected={key: observation(q) for key, q in dict(
                temperature=T+amplitude, perturbation=amplitude,
                sensibleEnergy=total_heat, modeHeat=mode_heat).items()}))
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(dict(precisionDigits=1600,
    method='Independent exponential field and analytic finite-window integral',
    referenceTemperature=273.15, cases=cases), indent=2, allow_nan=False)+'\n')
print(f'Wrote {len(cases)} independent cases to {path}')
