"""Independent high-precision Maxwell convolution and startup observations."""
import argparse
import json
from pathlib import Path
import mpmath as mp

mp.mp.dps = 800
parser = argparse.ArgumentParser()
parser.add_argument('--output', default='scripts/fixtures/continuum-startup-audit-independent.json')
path = Path(parser.parse_args().output)
if path.exists():
    raise SystemExit('Refusing to overwrite an existing reference')

def observation(q):
    if q == 0:
        return dict(value=0., sign=0, logMagnitude=None)
    _, exponent = mp.frexp(abs(q))
    quantum = max(-1074, exponent-53)
    scaled = mp.ldexp(abs(q), -quantum)
    n = int(mp.floor(scaled))
    remainder = scaled-n
    if remainder > mp.mpf('.5') or remainder == mp.mpf('.5') and n % 2:
        n += 1
    v = int(mp.sign(q))*float(n)*2.**quantum
    return dict(value=None if v == 0 else v, sign=int(mp.sign(q)),
                logMagnitude=float(mp.log(abs(q))))

settings = []
for E, eta in [(1.,1.), (2.,1.), (1.,2.), (2.,3.)]:
    for amplitude in [.001,-.001]:
        for omega in [.7,1.3]:
            settings.append(dict(elasticModulus=E,viscosity=eta,
                strainAmplitude=amplitude,angularFrequency=omega,
                initialElasticStrain=amplitude*omega*eta/E))
settings.append(dict(elasticModulus=2.,viscosity=1.,strainAmplitude=.002,
                     angularFrequency=1.,initialElasticStrain=.001))
cases = []
for p in settings:
    E,eta,a,w,e0 = map(mp.mpf,[p[k] for k in ['elasticModulus','viscosity',
        'strainAmplitude','angularFrequency','initialElasticStrain']])
    rate=E/eta
    def stress(t):
        # Closed evaluation of the convolution integral, independently of
        # the product implementation's small-time correction quadrature.
        return E*e0*mp.exp(-rate*t)+E*a*w*(rate*mp.cos(w*t)+w*mp.sin(w*t)-rate*mp.exp(-rate*t))/(rate*rate+w*w)
    energy=lambda t:stress(t)**2/(2*E)
    for time in [0.,1e-12,1e-20,1e-100,1e-120,.001]:
        t=mp.mpf(time)
        ds=E*(a*w-rate*e0) if t == 0 else mp.diff(stress,t)
        de=stress(0)*ds/E if t == 0 else mp.diff(energy,t)
        values=dict(stress=stress(t),stressRate=ds,energy=energy(t),
            energyRate=de,dissipation=stress(t)**2/eta,
            power=stress(t)*a*w*mp.cos(w*t),stressChange=stress(t)-stress(0),
            energyChange=energy(t)-energy(0))
        cases.append(dict(parameters=p,time=time,
            expected={k:observation(v) for k,v in values.items()}))
path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(dict(precisionDigits=800,
    method='Independent analytic convolution, differentiation and endpoint energy difference',
    cases=cases),indent=2,allow_nan=False)+'\n')
print(f'Wrote {len(cases)} independent startup cases')
