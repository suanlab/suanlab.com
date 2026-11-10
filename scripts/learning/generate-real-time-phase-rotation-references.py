#!/usr/bin/env python3
"""Offline mpmath diagnostic references, independent of production Machin/Taylor.

Requires mpmath for fixture regeneration only; Node tests use the checked-in
fixture. Agreement at two precisions is a diagnostic, not an interval proof.
"""
import json
import sys
from pathlib import Path
import mpmath as mp
sys.dont_write_bytecode = True
sys.set_int_max_str_digits(131072)
ROOT = Path(__file__).resolve().parents[2]

def binary_cell(value, bits):
    scaled = value * mp.mpf(2)**bits
    return {'lower': str(int(mp.floor(scaled))), 'upper': str(int(mp.ceil(scaled)))}

cases = []
for name, n, d, precision in [
    ('zero', 0, 1, 512), ('one', 1, 1, 512), ('negative-one', -1, 1, 512),
    ('tiny-positive', 1, 2**4096+1, 512), ('tiny-negative', -1, 2**4096+1, 512),
    ('rational-third', 1, 3, 512), ('negative-rational', -17, 7, 512),
    ('above-old-positive-limit', 2**64+1, 3, 512), ('above-old-negative-limit', -(2**64+1), 3, 512),
    ('phase-512-bits', 2**512+123, 999999999989, 512),
    ('phase-4096-bits', 2**4096+987654321, 999999999983, 512),
    ('phase-minus-8192-bits', -(2**8192+999999), 999999999979, 512),
    ('maximum-numerator', 2**65535+123456789, 1, 512),
    ('maximum-denominator', 17, 2**65535+123456789, 512),
    ('maximum-coprime-ratio', 2**65535+123456789, 2**65534+987654321, 512),
    ('precision-64', 2**4096+99999, 37, 64),
    ('precision-2048', -(2**4096+99999), 37, 2048),
]:
    magnitude = max(0, abs(n).bit_length()-d.bit_length()+1)
    reference_bits = precision+96
    results = []
    for guard in [192, 320]:
        with mp.workprec(magnitude+reference_bits+guard):
            x = mp.mpf(n)/d
            results.append({'sin': binary_cell(mp.sin(x), reference_bits), 'cos': binary_cell(mp.cos(x), reference_bits)})
    assert results[0] == results[1], name
    cases.append({'id':name,'numerator':str(n),'denominator':str(d),'precisionBits':precision,'referenceBits':reference_bits,**results[1]})
    print(name, flush=True)
# Points just to either side of a large approximate multiple of 2*pi.
with mp.workprec(2048):
    scale = 2**1024
    center = int(mp.floor(2*mp.pi*2**256*scale))
for delta in [-1, 1]:
    n, d, bits = center+delta, scale, 608
    results = []
    for guard in [192, 320]:
        with mp.workprec(3072+guard):
            x=mp.mpf(n)/d
            results.append({'sin':binary_cell(mp.sin(x),bits),'cos':binary_cell(mp.cos(x),bits)})
    assert results[0] == results[1]
    cases.append({'id':f'near-large-full-turn-{delta}','numerator':str(n),'denominator':str(d),'precisionBits':512,'referenceBits':bits,**results[1]})
with mp.workprec(1408):
    pi = binary_cell(mp.pi, 1152)
result={'source':'mpmath at two working precisions;diagnostic reference cells,not certified intervals','mpmathVersion':mp.__version__,'certifiedReference':False,'pi':{'referenceBits':1152,**pi},'cases':cases}
out=ROOT/'scripts/fixtures/real-time-phase-rotation-references.json'
out.write_text(json.dumps(result,indent=2)+'\n')
print('wrote',out,len(cases),'cases')
