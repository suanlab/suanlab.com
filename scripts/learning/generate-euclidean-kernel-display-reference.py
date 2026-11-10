"""Independent Decimal diagnostics; pi cancels from the kernel log ratios."""
from decimal import Decimal, localcontext
from fractions import Fraction as F
from pathlib import Path
import json
import runpy
import sys

sys.dont_write_bytecode = True
reference = runpy.run_path('scripts/learning/generate-euclidean-kernel-references.py')['reference']
inputs = json.loads(Path('scripts/fixtures/euclidean-kernel-reference-inputs.json').read_text())
selected = [row for row in inputs if row['id'] in ['one-variable-hand-check', 'off-diagonal-error-comparison', 'fractional-model', 'free-particle']]
default = inputs[0]['input']
selected += [
    dict(id='tiny-weak-potential', input={**default, 'omegaSquared': '1/999999999961', 'time': '1/999999999989'}),
    dict(id='large-log-unresolved', input={**default, 'omegaSquared': '1/999999999961', 'time': '1/999999999989', 'mass': '16', 'hbar': '1/999999999979', 'end': '16'}),
]


def dec(value):
    value = F(value)
    return Decimal(value.numerator) / Decimal(value.denominator)


with localcontext() as context:
    context.prec = 80
    cases = []
    for row in selected:
        source = row['input']
        tau, mass, hbar, omega2, a, b = [dec(source[k]) for k in ['time', 'mass', 'hbar', 'omegaSquared', 'start', 'end']]
        omega = omega2.sqrt()
        if omega:
            z = omega * tau
            sh, ch = (z.exp() - (-z).exp()) / 2, (z.exp() + (-z).exp()) / 2
            prefactor = mass * omega / (2 * hbar * sh)
            action = mass * omega * ((a * a + b * b) * ch - 2 * a * b) / (2 * hbar * sh)
            half_sh = ((z / 2).exp() - (-z / 2).exp()) / 2
            trace_log = -(2 * half_sh).ln()
        else:
            prefactor, action, trace_log = mass / (2 * hbar * tau), mass * (b - a) ** 2 / (2 * hbar * tau), None
        finite = reference(source)
        values = []
        for refinement in finite['refinements']:
            symmetric = (dec(refinement['prefactorSquaredTimesPi']) / prefactor).ln() / 2 - dec(refinement['minimumAction']) + action
            right = symmetric + dec(refinement['rightLogCorrection'])
            values.append(dict(intervals=refinement['intervals'], symmetricLogRatio=str(symmetric), rightLogRatio=str(right)))
        cases.append(dict(id=row['id'], input=source, continuousAction=str(action), continuousLogTrace=None if trace_log is None else str(trace_log), rows=values))
output = Path('scripts/fixtures/euclidean-kernel-display-reference.json')
output.write_text(json.dumps(dict(method='80-digit Decimal Mehler formula;Gaussian convolution finite reference;pi cancels', certifiedBounds=False, cases=cases), indent=2) + '\n')
print(json.dumps(dict(cases=len(cases), refinementRows=sum(len(c['rows']) for c in cases), bytes=output.stat().st_size)))
