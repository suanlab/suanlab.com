"""Independent normalized microstate oracle: Fraction inputs and mpmath.

The small Bregman contributions use positive curvature quadrature, not the
production finite series. Large contributions use the defining log expression.
Calculations use160/1000 digits; stored values retain80 significant digits.
"""
import argparse
import json
import math
import random
from fractions import Fraction as F
from pathlib import Path
import mpmath as mp

parser = argparse.ArgumentParser()
parser.add_argument('--output', default='scripts/fixtures/relative-entropy-independent.json')
out = Path(parser.parse_args().output)
if out.exists():
    raise SystemExit('Refusing to overwrite an existing reference')
rng = random.Random(26110651)
minimum, maximum = math.ulp(0.), float.fromhex('0x1.fffffffffffffp+1023')
kB = 1.380649e-23
scales = dict(identity=[], total=[64.], energy=[64., kB, 300.], rescue=[1e308, 1e308],
              subnormal=[minimum], zero=[0., maximum, maximum],
              overflowRescued=[maximum, maximum, minimum],
              underflowRescued=[minimum, minimum, 1e308, 1e308])


def number(v):
    return mp.mpf(v.numerator) / v.denominator if isinstance(v, F) else mp.mpf(v)


def projection(v):
    # Classify the binary64 projection BEFORE shortening the decimal string.
    # The exact stored mpf is a binary rational; Python's Fraction conversion
    # supplies a rounding path independent of the production JS implementation.
    sign, mantissa, exponent, _ = v._mpf_
    rational = F((-1 if sign else 1) * mantissa) * F(2)**exponent
    try:
        projected = float(rational)
    except OverflowError:
        projected = None
    if v != 0 and (projected == 0 or projected is None or not math.isfinite(projected)):
        projected = None
    return projected


def observation(v):
    fraction = v if isinstance(v, F) else None
    v = number(v)
    if v == 0:
        logarithm = None
    elif fraction is not None and abs(abs(fraction) - 1) <= F(1, 2):
        logarithm = mp.log1p(number(abs(fraction) - 1))
    else:
        logarithm = mp.log(abs(v))
    return dict(value=mp.nstr(v, 80), projected=projection(v), sign=int(mp.sign(v)),
                logMagnitude=None if logarithm is None else mp.nstr(logarithm, 80),
                logMagnitudeProjected=None if logarithm is None else projection(logarithm))


def case(w, v, digits, thermodynamic=False):
    with mp.workdps(digits):
        wf, vf = list(map(F, w)), list(map(F, v))
        p = [x / sum(wf) for x in wf]
        q = [x / sum(vf) for x in vf]
        coordinates, contributions = [], []
        hp, hq, cross = mp.mpf(0), mp.mpf(0), mp.mpf(0)
        for i, (pi, qi) in enumerate(zip(p, q)):
            dp = pi - qi
            pp, qq, gap = map(number, [pi, qi, dp])
            lp = None if pi == 0 else mp.log1p(number(pi - 1)) if pi >= F(1, 2) else mp.log(number(pi))
            lq = mp.log1p(number(qi - 1)) if qi >= F(1, 2) else mp.log(number(qi))
            log_ratio = None if pi == 0 else mp.log1p(number(dp / qi)) if abs(dp / qi) <= F(1, 2) else mp.log(pp / qq)
            if pi == 0:
                term = qq
            elif dp == 0:
                term = mp.mpf(0)
            elif abs(dp / qi) <= F(1, 2):
                # Integral of f''(z)=1/z on the straight p/q segment.
                term = gap**2 * mp.quad(lambda u: (1 - u) / (qq + u * gap), [0, 1])
            else:
                term = pp * mp.log(pp / qq) - gap
            pe = mp.mpf(0) if pi == 0 else -pp * lp
            qe = -qq * lq
            ce = -pp * lq
            assert term >= 0 and (term == 0) == (dp == 0)
            hp += pe
            hq += qe
            cross += ce
            contributions.append(term)
            coordinates.append(dict(index=i,
                regime='equal' if dp == 0 else 'zero-trial-probability' if pi == 0 else 'trial-above-target' if dp > 0 else 'trial-below-target',
                trialProbability=observation(pi), targetProbability=observation(qi),
                logTrialProbability=None if lp is None else observation(lp), logTargetProbability=observation(lq), difference=observation(dp),
                relativeDisplacement=observation(dp / qi), probabilityRatio=observation(pi / qi),
                logProbabilityRatio=None if log_ratio is None else observation(log_ratio),
                divergenceContribution=observation(term), trialEntropyContribution=observation(pe),
                targetEntropyContribution=observation(qe), crossEntropyContribution=observation(ce)))
        divergence = mp.fsum(contributions)
        same = p == q
        assert (divergence == 0) == same
        measures = dict(divergence=divergence, trialEntropy=hp, targetEntropy=hq, crossEntropy=cross)
        scaled = {}
        for name, values in scales.items():
            factor = F(1)
            for value in values:
                factor *= F(value)
            scaled[name] = {key: observation(value * number(factor)) for key, value in measures.items()}
        if thermodynamic:
            # Exact positive weights define a spectrum at this ONE temperature:
            # Ei=kB(100-300 ln vi). This checks the isothermal identity without
            # claiming a fixed-Hamiltonian temperature map has been implemented.
            energies = [number(F(kB)) * (100 - 300 * mp.log(number(x))) for x in vf]
            u = mp.fsum(number(pi) * energy for pi, energy in zip(p, energies))
            free = u - number(F(kB) * 300) * hp
            equilibrium = number(F(kB)) * (100 - 300 * mp.log(number(sum(vf))))
            residual = free - equilibrium - number(F(kB) * 300) * divergence
            assert abs(residual) < mp.power(10, -digits // 2) * max(number(F(kB)), abs(free), abs(equilibrium))
        return dict(trialWeights=w, targetWeights=v, scales=scales, digits=digits,
                    thermodynamicIdentityCheck=thermodynamic, sameDistribution=same,
                    measures={key: observation(value) for key, value in measures.items()},
                    coordinates=coordinates, scaledMeasures=scaled)


ordinary_inputs = [([1.], [2.]), ([1., 1.], [1., 1.]), ([1., 0.], [1., 1.]),
                   ([1., 2., 1.], [1., .5, .25]), ([1., 3., 7.], [2., 6., 14.]),
                   ([0., 1., 0.], [1., 2., 3.]), ([1., 1., 1.], [1., 2., 3.]),
                   ([1., 0., 2., 0.], [3., 2., 1., 4.])]
for count in [2, 3, 4, 8, 16, 32, 3, 5, 7, 9, 12, 20, 24, 32, 2, 6]:
    w = [10**rng.uniform(-6, 6) if rng.random() > .2 else 0. for _ in range(count)]
    if not any(w):
        w[0] = 1.
    v = [10**rng.uniform(-6, 6) for _ in range(count)]
    ordinary_inputs.append((w, v))
ordinary = [case(w, v, 160, i < 4) for i, (w, v) in enumerate(ordinary_inputs)]
up, down = math.nextafter(1., math.inf), math.nextafter(maximum, 0.)
limit_inputs = [
    ([1., 1.], [1., up]), ([1., up], [1., 1.]),
    ([1., 1., 1.], [1., 1., up]), ([1., 1., minimum], [up, up, minimum]),
    ([maximum, maximum, minimum], [down, down, minimum]),
    ([1., minimum], [1., 2 * minimum]),
    ([maximum, minimum], [maximum, 2 * minimum]),
    ([1., minimum], [minimum, 1.]), ([maximum, minimum], [minimum, maximum]),
    ([0., 1.], [minimum, 1.]), ([0., maximum], [minimum, maximum]),
    ([minimum] * 32, [maximum] * 32), ([maximum] + [0.] * 31, [maximum] * 32),
    ([minimum] * 16 + [0.] * 16, [minimum] * 32),
    ([9., 7.], [1., 1.]), ([math.nextafter(9., 0.), 7.], [1., 1.]),
    ([math.nextafter(9., math.inf), 7.], [1., 1.]),
    ([minimum, 0.], [minimum, minimum]),
    ([maximum, maximum], [maximum, down]),
    ([minimum, 2 * minimum, 3 * minimum], [maximum / 3, 2 * (maximum / 3), maximum]),
]
limits = [case(w, v, 1000) for w, v in limit_inputs]
assert len(ordinary) == 24 and len(limits) == 20
out.write_text(json.dumps(dict(
    method='Exact Fraction normalization; independent positive-curvature quadrature near equality and direct log definition away; four reference-temperature, weight-defined-spectrum identity checks.',
    serializedSignificantDigits=80, ordinary=ordinary, limits=limits), indent=2) + '\n')
print(f'Wrote {len(ordinary)} ordinary160-digit and {len(limits)} limit1000-digit cases to {out}')
