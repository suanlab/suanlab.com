"""Independent Fraction channel laws, direct high-precision entropy and signed MI.

Precision derives from every returned probability law and squared joint displacements,
then each case is recomputed at 80 additional decimal digits. No production code
or positive Bregman production formula is used to evaluate reference MI.
"""
import argparse
import json
import math
import random
from fractions import Fraction as F
from pathlib import Path
import mpmath as mp

parser = argparse.ArgumentParser()
parser.add_argument('--output', default='scripts/fixtures/noisy-macro-observation-independent.json')
out = Path(parser.parse_args().output)
if out.exists():
    raise SystemExit('Refusing to overwrite reference')

def normalized(values):
    values = list(map(F, values))
    return [v / sum(values) for v in values]

def m(v):
    return mp.mpf(v.numerator) / v.denominator if isinstance(v, F) else mp.mpf(v)

def obs(v):
    v = m(v); linear = float(v)
    if v and (linear == 0 or not math.isfinite(linear)):
        linear = None
    return dict(value=linear, sign=int(mp.sign(v)), logMagnitude=float(mp.log(abs(v))) if v else None)

def entropy(values):
    return -mp.fsum(m(p) * mp.log(m(p)) for p in values if p)

def case(name, weights, rows):
    p = normalized(weights); channel = list(map(normalized, rows)); n = len(p); width = len(rows[0])
    joint = [[p[i] * channel[i][j] for j in range(width)] for i in range(n)]
    marginal = [sum((joint[i][j] for i in range(n)), F(0)) for j in range(width)]
    posterior = [[joint[i][j] / marginal[j] for i in range(n)] if marginal[j] else None for j in range(width)]
    target = [[p[i] * marginal[j] for j in range(width)] for i in range(n)]
    assert sum(p) == 1 and sum(marginal) == 1
    assert all(sum(row) == 1 for row in channel)
    assert all(row is None or sum(row) == 1 for row in posterior)
    independent = joint == target
    scales = [v for row in joint + channel + [p, marginal] + [r for r in posterior if r is not None] for v in row if v]
    for i in range(n):
        for j in range(width):
            q = target[i][j]
            if q:
                displacement = (joint[i][j] - q) / q
                if displacement:
                    scales.append(q * displacement * displacement)
    # Bit lengths give a conservative logarithmic scale estimate to within one
    # decimal digit. 160 guard digits absorb direct-sum cancellation; the second
    # independent precision pass checks the actually serialized observations.
    digits = max([160] + [math.ceil(max(0, v.denominator.bit_length() - abs(v.numerator).bit_length()) * math.log10(2)) + 160 for v in scales])
    def calculate(precision):
        with mp.workdps(precision):
            hx = entropy(p); hy = entropy(marginal); hxy = entropy([v for row in joint for v in row])
            hrow = [entropy(row) for row in channel]
            hpost = [entropy(row) if row is not None else None for row in posterior]
            noise = [m(p[i]) * hrow[i] for i in range(n)]
            hidden = [m(marginal[j]) * hpost[j] if marginal[j] else mp.mpf(0) for j in range(width)]
            mi = mp.mpf(0) if independent else mp.fsum(m(joint[i][j]) * mp.log(m(joint[i][j] / target[i][j])) for i in range(n) for j in range(width) if joint[i][j])
            tolerance = mp.power(10, -(precision - 40))
            assert abs(hxy - hx - mp.fsum(noise)) < tolerance
            assert abs(hxy - hy - mp.fsum(hidden)) < tolerance
            assert abs(mi - hx + mp.fsum(hidden)) < tolerance
            assert mi >= 0 and mi <= min(hx, hy) + tolerance
            return dict(inputProbabilities=list(map(obs, p)), channelProbabilities=[list(map(obs, row)) for row in channel],
                jointProbabilities=[list(map(obs, row)) for row in joint], outputProbabilities=list(map(obs, marginal)),
                posteriorProbabilities=[None if row is None else list(map(obs, row)) for row in posterior],
                rowEntropies=list(map(obs, hrow)), posteriorEntropies=[None if v is None else obs(v) for v in hpost],
                noiseEntropyContributions=list(map(obs, noise)), hiddenEntropyContributions=list(map(obs, hidden)),
                inputEntropy=obs(hx), outputEntropy=obs(hy), jointEntropy=obs(hxy), noiseConditionalEntropy=obs(mp.fsum(noise)),
                hiddenConditionalEntropy=obs(mp.fsum(hidden)), mutualInformation=obs(mi))
    result = calculate(digits)
    assert result == calculate(digits + 80), name
    return dict(name=name, preparationWeights=weights, channelWeights=rows, calculationDigits=digits, confirmationDigits=digits+80,
        independentExactly=independent, deterministicOnPreparationSupport=all(not p[i] or sum(bool(x) for x in channel[i]) == 1 for i in range(n)), **result)

ordinary_inputs = [
 ('identity',[3,1],[[1,0],[0,1]]),
 ('deterministic-partition',[6,2,1,1],[[1,0],[1,0],[0,1],[0,1]]),
 ('noise-over-deterministic-input',[1,0],[[1,1],[1,1]]),
 ('independent-noise',[3,1],[[1,1],[7,7]]),
 ('binary-symmetric-error',[1,1],[[9,1],[1,9]]),
 ('erasure',[1,1],[[3,0,1],[0,3,1]]),
 ('unused-output',[1,2],[[1,1,0],[1,0,0]]),
 ('unused-input',[1,0],[[1,3],[7,0]]),
 ('invertible-flip',[3,1],[[0,1],[1,0]]),
 ('single-input-output',[7],[[9]]),
]
rng = random.Random(26110631)
for j in range(14):
    n = rng.randint(2,5); width = rng.randint(2,5)
    w = [rng.randint(0,12) for _ in range(n)]; w[0] += 1
    rows = [[rng.randint(0,12) for _ in range(width)] for _ in range(n)]
    for row in rows:row[0] += 1
    ordinary_inputs.append((f'integer-{j}',w,rows))
small = math.ulp(0.); large = float.fromhex('0x1.fffffffffffffp+1023'); next_one = math.nextafter(1.,2.)
limiting_inputs = [
 ('overflowing-prior-and-rows',[large,large],[[large,large],[large,large/2]]),
 ('subnormal-only',[small,2*small],[[small,3*small],[2*small,small]]),
 ('underflowing-input',[small,large],[[1,0],[0,1]]),
 ('underflowing-transition',[1,1],[[small,large],[large,small]]),
 ('double-underflowing-joint',[small,large],[[small,large],[large,small]]),
 ('one-ulp-near-independent',[1,1],[[1,next_one],[1,1]]),
 ('tiny-channel-difference',[1,1],[[large,small],[large,2*small]]),
 ('tiny-prior-and-channel-difference',[small,large],[[large,small],[large,2*small]]),
 ('hidden-normalization-difference',[1,1],[[large,small,small],[large,2*small,small]]),
 ('unobserved-noisy-row',[1,0],[[1,0],[small,large]]),
 ('independent-extreme-rows',[small,large],[[large,small],[large,small]]),
 ('rare-output-empty-column',[large,small],[[large,small,0],[large,2*small,0]]),
]
result = dict(method='Exact Fraction channel normalization; direct signed high-precision MI sum; precision selected from all returned probability laws and exact joint/target displacement scales and every serialized observation confirmed at 80 extra decimal digits.',
    ordinary=[case(*v) for v in ordinary_inputs], limits=[case(*v) for v in limiting_inputs])
out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print('Wrote24 ordinary and12 limiting cases; max base digits',max(c['calculationDigits'] for c in result['ordinary']+result['limits']))
