"""Regenerate independent CTMC entropy references; requires sympy and mpmath.

Reads preparations only, never a production result or implementation. Stationarity
comes from a symbolic nullspace; moments from direct unshifted 3x3 exponentials.
Keep existing fixtures immutable: pass a new --output path, then compare results.
"""
import argparse
import json
from pathlib import Path
import sympy as s
import mpmath as mp

parser = argparse.ArgumentParser()
parser.add_argument('--preparations', default='scripts/fixtures/entropy-production-preparation.json')
parser.add_argument('--output', required=True)
args = parser.parse_args()
output = Path(args.output)
if output.exists():
    raise SystemExit('Refusing to overwrite an existing reference')
preparations = json.loads(Path(args.preparations).read_text())
if isinstance(preparations, dict):
    preparations = preparations['cases']
mp.mp.dps = 220
backgrounds, moments, fixtures = {}, {}, []

def mm(x):
    return mp.mpf(str(s.numer(x))) / mp.mpf(str(s.denom(x)))

for preparation in preparations:
    p = preparation['input']
    rates = tuple(s.Rational(p[k], p['rateDenominator']) for k in ['a', 'b', 'c', 'd', 'e', 'f'])
    if rates not in backgrounds:
        a, b, c, d, e, f = rates
        Q = s.Matrix([[-a-f, a, f], [d, -d-b, b], [c, e, -c-e]])
        v = Q.T.nullspace()[0]
        pi = (v / sum(v)).T
        qm = mp.matrix([[mm(Q[i, j]) for j in range(3)] for i in range(3)])
        pm = mp.matrix([[mm(x) for x in pi]])
        flow = lambda i, j: pi[i] * Q[i, j]
        sigma = sum(mm(flow(i, j)-flow(j, i)) * mp.log(mm(flow(i, j)/flow(j, i))) for i, j in [(0, 1), (1, 2), (2, 0)])
        affinity = mp.log(mm(a*b*c / (d*e*f)))
        current = flow(0, 1) - flow(1, 0)
        backgrounds[rates] = Q, pi, qm, pm, sigma, affinity, current
    Q, pi, qm, pm, sigma, affinity, current = backgrounds[rates]
    states = [p['initialState']]
    for k in range(p['jumpCount']):
        states.append((states[-1] + (1 if not (p['pathCode'] >> k) & 1 else -1)) % 3)
    ratio = pi[states[0]] / pi[states[-1]]
    for i, j in zip(states, states[1:]):
        ratio *= Q[i, j] / Q[j, i]
    ratio = s.factor(ratio)
    z = s.Rational(p['tiltNumerator'], p['tiltDenominator'])
    L = mp.matrix([[qm[i, j] if i == j else qm[i, j] * mp.exp(-mm(z) * mp.log(mm(pi[i]*Q[i, j] / (pi[j]*Q[j, i])))) for j in range(3)] for i in range(3)])
    rows = []
    for step in range(p['sampleSteps'] + 1):
        t = s.Rational(p['timeNumerator'] * step, p['timeDenominator'] * p['sampleSteps'])
        key = rates, t, z
        if key not in moments:
            value = (pm * mp.expm(L * mm(t)) * mp.matrix([1, 1, 1]))[0]
            assert value > 0
            moments[key] = mp.log(value)
        rows.append(dict(time=str(t), meanEntropy=mp.nstr(mm(t)*sigma, 180), logMoment=mp.nstr(moments[key], 180)))
    fixtures.append(dict(label=preparation['label'], input=p, stationary=[str(x) for x in pi], current=str(current), affinity=mp.nstr(affinity, 180), entropyRate=mp.nstr(sigma, 180), pathTotal=mp.nstr(mp.log(mm(ratio)), 180), pathSign=0 if ratio == 1 else 1 if ratio > 1 else -1, rows=rows))

output.write_text(json.dumps(dict(provenance='Independent SymPy nullspace stationarity and220-digit direct mpmath exponentials; stored180digits. Regression comparison uncertainty1e-160, not a production tolerance.', cases=fixtures), indent=2) + '\n')
print(json.dumps(dict(preparations=len(fixtures), timeRecords=sum(len(c['rows']) for c in fixtures), digits=mp.mp.dps, output=str(output))))
