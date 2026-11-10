"""Independent Decimal canonical binary state sums and logarithmic bounds."""
import argparse
from decimal import Decimal as D, localcontext
import json
import math
from pathlib import Path
import sys

KB = 1.380649e-23


def exact(x):
    return D.from_float(float(x))


def number(x):
    if not x:
        return '0'
    value = float(x)
    return None if not math.isfinite(value) or value == 0 and x != 0 else format(x, '.79e')


def zero():
    return dict(value='0', sign=0, logMagnitude=None, logNegativeLogMagnitude=None,
                logMagnitudeStatus='not-applicable', logNegativeLogMagnitudeStatus='not-applicable')


def from_log(log, probability=False):
    value = number(log.exp()) if -1000 < log < 1000 else None
    # Moment logarithms are approximate near one. Probability logs must retain
    # positive loss even when their first logarithm is below binary64 range.
    first = number(log) if probability else ('0' if not log else format(log, '.79e')) if math.isfinite(float(log)) else None
    nested = number((-log).ln()) if log < 0 else None
    return dict(value=value, sign=1, logMagnitude=first,
                logNegativeLogMagnitude=nested,
                logMagnitudeStatus='finite' if first is not None else 'outside-number-range',
                logNegativeLogMagnitudeStatus='not-applicable' if log >= 0 else 'finite' if nested is not None else 'outside-number-range')


def observed(x):
    return zero() if not x else from_log(x.ln())


def log1p(x):
    if x > D('.125'):
        return (1 + x).ln()
    total, power, index = x, x, 1
    while True:
        index += 1
        power *= -x
        next_value = total + power / index
        if next_value == total:
            return total
        total = next_value


def near_one(log_loss):
    return dict(value='1', sign=1, logMagnitude=None, logMagnitudeStatus='outside-number-range',
                logNegativeLogMagnitude=number(log_loss),
                logNegativeLogMagnitudeStatus='finite' if number(log_loss) is not None else 'outside-number-range')


def compute(p, precision):
    with localcontext() as c:
        c.prec = precision
        n, epsilon, temperature = p['sites'], exact(p['energyGap']), exact(p['temperature'])
        kb = exact(KB)
        cold = temperature == 0 and epsilon > 0
        negative = temperature < 0
        u = D(0) if epsilon == 0 or cold else epsilon / (kb * abs(temperature))
        distribution = []
        if cold:
            a, z = zero(), observed(D(1))
            distribution = [dict(excitations=k, multiplicity=str(math.comb(n, k)), probability=z if k == 0 else a) for k in range(n + 1)]
            expected = {key: a for key in ['meanExcitations', 'excitationVariance', 'meanEnergy', 'energyVariance', 'entropy', 'entropyInKB', 'heatCapacity', 'heatCapacityInKB', 'logPartition']}
            expected.update(oneProbability=a, zeroProbability=z, relativeCountFluctuation=None, relativeEnergyFluctuation=None)
        elif u > 10000:
            # log1p(exp(-u)) differs from exp(-u) by relative < exp(-u)/2.
            # At this cutoff, even the largest SI prefactors cannot make the
            # omitted correction observable in binary64. No false support zero.
            a, z = from_log(-u, True), near_one(-u)
            for k in range(n + 1):
                j = n - k if negative else k
                probability = observed(D(1)) if n == 0 else near_one(D(n).ln() - u) if j == 0 else from_log(D(math.comb(n, k)).ln() - j * u, True)
                distribution.append(dict(excitations=k, multiplicity=str(math.comb(n, k)), probability=probability))
            q = lambda scale: zero() if n == 0 else from_log(D(n).ln() + scale - u)
            expected = dict(oneProbability=z if negative else a, zeroProbability=a if negative else z,
                            meanExcitations=observed(D(n)) if negative else q(D(0)), excitationVariance=q(D(0)),
                            meanEnergy=observed(n * epsilon) if negative else q(epsilon.ln()), energyVariance=q(2 * epsilon.ln()),
                            entropy=q(kb.ln() + (1 + u).ln()), entropyInKB=q((1 + u).ln()),
                            heatCapacity=q(kb.ln() + 2 * u.ln()), heatCapacityInKB=q(2 * u.ln()),
                            logPartition=observed(n * u) if negative else q(D(0)), relativeCountFluctuation=None if n == 0 else from_log(((-u if negative else u) - D(n).ln()) / 2),
                            relativeEnergyFluctuation=None if n == 0 else from_log(((-u if negative else u) - D(n).ln()) / 2))
        else:
            w = (-u).exp()
            W, correction = 1 + w, log1p(w)
            a, z = from_log(-u - correction, True), from_log(-correction, True)
            for k in range(n + 1):
                distribution.append(dict(excitations=k, multiplicity=str(math.comb(n, k)), probability=from_log(D(math.comb(n, k)).ln() - (n - k if negative else k) * u - n * correction, True)))
            mean, variance = n * (1 if negative else w) / W, n * w / (W * W)
            entropy = n * (correction + u * w / W)
            capacity = n * u * u * w / (W * W)
            expected = dict(oneProbability=z if negative else a, zeroProbability=a if negative else z,
                            meanExcitations=observed(mean), excitationVariance=observed(variance),
                            meanEnergy=observed(epsilon * mean), energyVariance=observed(epsilon * epsilon * variance),
                            entropy=observed(kb * entropy), entropyInKB=observed(entropy),
                            heatCapacity=observed(kb * capacity), heatCapacityInKB=observed(capacity), logPartition=observed(n * (u + correction) if negative else n * correction),
                            relativeCountFluctuation=None if n == 0 else from_log(((-u if negative else u) - D(n).ln()) / 2),
                            relativeEnergyFluctuation=None if n == 0 or epsilon == 0 else from_log(((-u if negative else u) - D(n).ln()) / 2))
            # Direct sums over K independently check analytic moments. At very
            # cold inputs, stable positive terms avoid cancellation of variance.
            probabilities = [D(math.comb(n, k)) * w ** (n - k if negative else k) / W ** n for k in range(n + 1)]
            assert abs(sum(probabilities) - 1) < D(10) ** (-precision + 8)
            first = sum(k * prob for k, prob in enumerate(probabilities))
            centered = sum((D(k) - mean) ** 2 * prob for k, prob in enumerate(probabilities))
            for x, y in [(first, mean), (centered, variance)]:
                assert abs(x - y) <= max(abs(y), D(1)) * D(10) ** (-precision + 10)
        return dict(expected=expected, distribution=distribution,
                    method='analytic logarithmic bounds beyond u=10000' if u > 10000 else 'Decimal direct count sums and analytic moments' if not cold else 'exact zero-temperature preparation')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('scripts/fixtures/canonical-binary-independent.json'))
    args = parser.parse_args()
    inputs = [(1, KB, 1), (2, KB * math.log(2), 1), (4, KB, 1), (8, KB * 300, 300),
              (16, KB * 600, 300), (32, KB * 12, 300), (8, KB * 300, 30), (4, KB * 30, 300),
              (1, 0, 1), (2, 0, 300), (4, 0, 0), (0, 0, 0), (0, KB, 1), (4, KB, 0),
              (1, KB * 700, 1), (2, KB * 750, 1), (4, KB * 1000, 1), (1, KB * 2100, 1),
              (4, KB * 4096, 1), (4, KB * 4097, 1), (2, 1e280, 1e280 / (KB * 800)),
              (4, 1e280, 1e280 / (KB * 1000)), (4, 1, sys.float_info.min),
              (4, 1, math.ulp(0.0)), (4, sys.float_info.max, math.ulp(0.0)),
              (2, math.ulp(0.0), sys.float_info.max), (1, math.ulp(0.0), 1),
              (2, 4e261, 1e-24), (4, 1e-300, 1e300), (2, 1e308, 1e308), (1, KB, -1), (4, KB, -1), (8, KB * 300, -300),
              (2, KB * 750, -1), (4, KB * 4097, -1), (4, 1, -math.ulp(0.0)),
              (2, math.ulp(0.0), -sys.float_info.max), (4, 0, -1)]
    cases = []
    for n, gap, temperature in inputs:
        p = dict(sites=n, energyGap=gap, temperature=temperature)
        with localcontext() as c:
            c.prec = 80
            u = exact(gap) / (exact(KB) * abs(exact(temperature))) if gap and temperature else D(0)
            precision = max(220, 2 * int(-u.log10()) + 120) if u else 220
        low, high = compute(p, precision), compute(p, precision + 80)
        # Compare the exported 80-digit observations at two working precisions.
        assert low == high, (p, precision)
        cases.append(dict(parameters=p, precisionDigits=[precision, precision + 80], **high))
    payload = dict(boltzmannConstantBinary=KB, scope='Independent references only; teaching UI and registration pending.', cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(dict(cases=len(cases), output=str(args.output))))


if __name__ == '__main__':
    main()
