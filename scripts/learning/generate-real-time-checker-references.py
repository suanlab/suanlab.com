"""Decimal diagnostics at two precisions, not certified reference intervals."""
import json
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from pathlib import Path

BITS = 1280
CASES = [
    ('one', '1', '1'), ('half', '1', '2'), ('third', '1', '3'),
    ('two', '2', '1'), ('sixteen', '16', '1'),
    ('tiny', '1', '10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'),
    ('large', '10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', '1'),
    ('mixed', '999999999983', '999999999989'),
    ('large-decay', '6000', '1'), ('moderate-decay', '5630', '1'),
]


def evaluate(n, d, precision, bits=BITS):
    with localcontext() as context:
        context.prec = precision
        x = Decimal(n) / Decimal(d)
        values = {'log': x.ln(), 'expNegative': (-x).exp()}
        scale = Decimal(2) ** bits
        result = {name: {'lower': str(int((v * scale).to_integral_value(rounding=ROUND_FLOOR))),
                         'upper': str(int((v * scale).to_integral_value(rounding=ROUND_CEILING)))}
                  for name, v in values.items()}
        if values['expNegative'] == 0:
            # Decimal's exponent range can underflow. A positive exponential
            # must not become a purported exact-zero diagnostic cell.
            assert x >= bits  # e>2 implies exp(-x)<2^-bits.
            result['expNegative'] = dict(lower='0', upper='1')
        return result


rows = []
for name, n, d in CASES:
    first, second = evaluate(n, d, 700), evaluate(n, d, 900)
    assert first == second, name
    row = dict(id=name, numerator=n, denominator=d, **first)
    if name in ('large-decay', 'moderate-decay'):
        a = evaluate(n, d, 4000, 10240)['expNegative']
        b = evaluate(n, d, 4300, 10240)['expNegative']
        assert a == b, name
        row['highPrecisionExpNegative'] = dict(referenceBits=10240, **a)
    rows.append(row)
output = dict(source='Python Decimal at700/900decimal digits;diagnostic cells,not certified intervals',
              certifiedReference=False, referenceBits=BITS,
              decimalUnderflowHandling='positive exponential underflow replaced by[0,1]using x>=referenceBits ande>2', cases=rows)
Path('scripts/fixtures/real-time-checker-transcendental-references.json').write_text(
    json.dumps(output, indent=2) + '\n')
