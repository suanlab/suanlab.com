"""Exact Fraction polynomial differentiation/integration of the thermoelectric bar.
No product imports. Binary64 inputs are converted by their exact integer ratios.
"""
import argparse
import json
import math
from decimal import Decimal, localcontext
from fractions import Fraction as F
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--output', default='scripts/fixtures/continuum-electric-audit-independent.json')
output = Path(parser.parse_args().output)
if output.exists():
    raise SystemExit('Refusing to overwrite reference')

def plus(a, b):
    return [(a[i] if i < len(a) else F(0)) + (b[i] if i < len(b) else F(0)) for i in range(max(len(a), len(b)))]

def scale(a, v):
    return [q * v for q in a]

def derivative(a):
    return [i * a[i] for i in range(1, len(a))]

def at(a, x):
    return sum(q * x**i for i, q in enumerate(a))

def integral(a, x):
    return sum(q * x**(i+1) / (i+1) for i, q in enumerate(a))

def observation(q):
    if q == 0:
        return dict(value=0, sign=0, logMagnitude=None)
    try:
        value = float(q)
    except OverflowError:
        value = math.inf
    with localcontext() as context:
        context.prec = 100
        logarithm = (Decimal(abs(q.numerator)) / Decimal(q.denominator)).ln()
    return dict(value=value if value != 0 and math.isfinite(value) else None,
                sign=1 if q > 0 else -1, logMagnitude=str(logarithm))

base = dict(length=1, area=1, resistivity=1, conductivity=1, seebeck=.01,
            leftTemperature=300, rightTemperature=310, currentDensity=-.1, referencePotential=0)
rows = []
def case(label, changes=None, xi=.5):
    parameters = dict(base, **(changes or {}))
    L,A,r,K,S,TL,TR,j,V0 = [F(parameters[k]) for k in ['length','area','resistivity','conductivity','seebeck','leftTemperature','rightTemperature','currentDensity','referencePotential']]
    x = F(xi)
    bulge = r*j*j*L*L/(2*K)
    temperature = [TL, TR-TL+bulge, -bulge]
    potential = plus([V0, -r*j*L], scale(plus(temperature, [-TL]), -S))
    field = scale(derivative(potential), -1/L)
    heat = plus(scale(temperature, S*j), scale(derivative(temperature), -K/L))
    work = scale(field, j)
    heat_gradient = scale(derivative(heat), 1/L)
    # Independent polynomial differentiation verifies local energy conservation.
    assert work == heat_gradient
    pin = A*L*integral(work, F(1))
    ql,qr = A*at(heat,F(0)),A*at(heat,F(1))
    assert pin == qr-ql
    drop = at(potential,F(0))-at(potential,F(1))
    assert pin == A*j*drop
    expected = dict(temperature=at(temperature,x), temperatureGradient=at(derivative(temperature),x)/L,
                    temperatureCurvature=at(derivative(derivative(temperature)),x)/(L*L),
                    electricField=at(field,x), potential=at(potential,x), stablePotentialChange=at(potential,x)-V0,
                    terminalVoltageDrop=drop, totalHeatFlux=at(heat,x),
                    localElectricalWork=at(work,x), heatFluxGradient=at(heat_gradient,x),
                    leftHeatInput=ql, rightHeatOutput=qr, electricalPowerInput=pin,
                    stableBoundaryHeatDifference=qr-ql, partialElectricalPower=A*L*integral(work,x),
                    stableJouleTemperatureRise=at(temperature,x)-(TL+(TR-TL)*x),
                    joulePower=A*r*j*j*L, reversibleElectricalPower=A*S*j*(TR-TL))
    rows.append(dict(label=label, parameters=parameters, xi=xi,
                     expected={k:observation(v) for k,v in expected.items()},
                     leftCoolingCOP=observation(ql/pin) if ql>0 and pin>0 else None,
                     electricalRegime='generation' if pin<0 else 'zero-terminal-work' if pin==0 else 'electrical-input'))

for direction in [-math.inf, math.inf]:
    current = -.1
    for step in range(6):
        case(f'near-terminal-cancellation-{direction}-{step}', {'currentDensity':current})
        current = math.nextafter(current,direction)
for x in [0, .125, .5, .875, 1]:
    case(f'exact-dyadic-zero-{x}',dict(seebeck=.015625,rightTemperature=308,currentDensity=-.125),x)
    case(f'open-circuit-{x}',dict(currentDensity=0),x)
    case(f'tiny-current-{x}',dict(currentDensity=math.ulp(0.0),rightTemperature=300),x)
for s in [0,math.ulp(0.0),-math.ulp(0.0),1e-20,-1e-20]:
    case(f'left-heat-cancellation-{s}',dict(seebeck=s,rightTemperature=299.5,currentDensity=1))
for j in [-10,-.05,.9,10]:
    case(f'ordinary-{j}',dict(currentDensity=j,length=1.3,area=.7,referencePotential=4),.37123456789)
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps(dict(method='exact Fraction polynomial fields; independent physical derivatives and integrals; 100-digit log', cases=rows),indent=2)+'\n')
print(f'{len(rows)} independent rational thermoelectric cases written')
