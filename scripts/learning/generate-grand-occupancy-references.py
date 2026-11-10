"""Independent Fraction input differences, Decimal sums, and extreme log bounds."""
import argparse
from decimal import Decimal as D, localcontext
from fractions import Fraction
import importlib.util
from itertools import product
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True

spec = importlib.util.spec_from_file_location('canonical_reference_helpers', Path(__file__).with_name('generate-canonical-binary-references.py'))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
KB = helpers.KB
zero, from_log, log1p, near_one = helpers.zero, helpers.from_log, helpers.log1p, helpers.near_one


def dec(f):
    return D(f.numerator) / D(f.denominator)


def observed(x):
    if not x:
        return zero()
    q = helpers.from_log(x.ln())
    value = float(x)
    q['value'] = repr(value) if math.isfinite(value) and value != 0 else None
    return q


def fraction_linear(f):
    try:
        value = float(f)
    except OverflowError:
        return None
    return repr(value) if math.isfinite(value) and (value != 0 or f == 0) else None


def signed(q, sign):
    if q['sign'] == 0:
        return q
    out = dict(q, sign=sign)
    if out['value'] is not None and sign < 0:
        out['value'] = '-' + out['value']
    return out


def signed_observed(x):
    return signed(observed(abs(x)), -1 if x < 0 else 1)


def evaluate(p, precision):
    with localcontext() as c:
        c.prec = precision
        M = p['sites']
        epsilon, mu, T, kb = [D.from_float(float(x)) for x in [p['energyGap'], p['chemicalPotential'], p['temperature'], KB]]
        delta_fraction = Fraction.from_float(p['energyGap']) - Fraction.from_float(p['chemicalPotential'])
        delta = dec(delta_fraction)
        cold, sign_t = T == 0, -1 if T < 0 else 1
        theta_fraction = Fraction(1) if cold else Fraction.from_float(KB) * abs(Fraction.from_float(p['temperature']))
        theta = dec(theta_fraction)
        y = D(0) if cold else dec(-delta_fraction * sign_t / theta_fraction)
        u, positive = abs(y), y > 0
        distribution, enumerated = [], False
        if cold and delta:
            particles = M if delta < 0 else 0
            for k in range(M + 1):
                probability = observed(D(1)) if k == particles else zero()
                distribution.append(dict(particles=k, multiplicity=str(math.comb(M, k)), probability=probability,
                                         conditionalPatternProbability=observed(D(1)) if k == particles else None))
            expected = {key: zero() for key in ['numberVariance', 'energyVariance', 'energyNumberCovariance', 'entropy', 'entropyInKB', 'numberSusceptibility']}
            expected.update(oneProbability=observed(D(1)) if delta < 0 else zero(), zeroProbability=zero() if delta < 0 else observed(D(1)),
                            meanParticles=observed(D(particles)), meanHoles=observed(D(M - particles)), meanEnergy=observed(epsilon * particles),
                            relativeNumberFluctuation=zero() if particles > 0 else None,
                            grandPotential=signed_observed(delta * particles), logGrandPartition=None if delta < 0 and M else zero())
            log_status = 'infinite-zero-temperature-limit' if delta < 0 and M else 'finite-zero-temperature-limit'
            susceptibility_status = 'empty-system' if M == 0 else 'zero-temperature-away-from-threshold'
        elif u > 10000:
            minor, major = from_log(-u, True), near_one(-u)
            q = lambda constant: zero() if M == 0 else from_log(D(M).ln() + constant - u)
            for k in range(M + 1):
                j = M - k if positive else k
                probability = observed(D(1)) if M == 0 else near_one(D(M).ln() - u) if j == 0 else from_log(D(math.comb(M, k)).ln() - j * u, True)
                distribution.append(dict(particles=k, multiplicity=str(math.comb(M, k)), probability=probability,
                                         conditionalPatternProbability=observed(D(1) / math.comb(M, k))))
            potential = observed(M * abs(delta)) if positive else q(theta.ln())
            expected = dict(oneProbability=major if positive else minor, zeroProbability=minor if positive else major,
                            meanParticles=observed(D(M)) if positive else q(D(0)), meanHoles=q(D(0)) if positive else observed(D(M)),
                            numberVariance=q(D(0)), meanEnergy=zero() if not epsilon else observed(M * epsilon) if positive else q(epsilon.ln()),
                            energyVariance=zero() if not epsilon else q(2 * epsilon.ln()), energyNumberCovariance=zero() if not epsilon else q(epsilon.ln()),
                            entropy=q(kb.ln() + (1 + u).ln()), entropyInKB=q((1 + u).ln()),
                            numberSusceptibility=signed(q(-theta.ln()), sign_t), relativeNumberFluctuation=None if M == 0 else from_log((-y - D(M).ln()) / 2),
                            grandPotential=signed(potential, -sign_t), logGrandPartition=observed(M * u) if positive else q(D(0)))
            log_status, susceptibility_status = 'finite-temperature', 'empty-system' if M == 0 else 'finite-temperature'
        else:
            w = (-u).exp()
            W, correction = 1 + w, log1p(w)
            minor, major = from_log(-u - correction, True), from_log(-correction, True)
            probabilities = []
            for k in range(M + 1):
                j = M - k if positive else k
                probability = from_log(D(math.comb(M, k)).ln() - j * u - M * correction, True)
                probabilities.append(D(math.comb(M, k)) * w ** j / W ** M)
                distribution.append(dict(particles=k, multiplicity=str(math.comb(M, k)), probability=probability,
                                         conditionalPatternProbability=observed(D(1) / math.comb(M, k))))
            mean, holes = M * (1 if positive else w) / W, M * (w if positive else 1) / W
            variance, entropy = M * w / W ** 2, M * (correction + u * w / W)
            log_xi = M * ((y if positive else 0) + correction)
            expected = dict(oneProbability=major if positive else minor, zeroProbability=minor if positive else major,
                            meanParticles=observed(mean), meanHoles=observed(holes), numberVariance=observed(variance),
                            meanEnergy=observed(epsilon * mean), energyVariance=observed(epsilon ** 2 * variance), energyNumberCovariance=observed(epsilon * variance),
                            entropy=observed(kb * entropy), entropyInKB=observed(entropy),
                            numberSusceptibility=None if cold and M else signed(observed(variance / theta), sign_t),
                            relativeNumberFluctuation=None if M == 0 else from_log((-y - D(M).ln()) / 2),
                            grandPotential=zero() if cold else signed_observed(-sign_t * theta * log_xi), logGrandPartition=observed(log_xi))
            log_status = 'finite-zero-temperature-limit' if cold else 'finite-temperature'
            susceptibility_status = 'empty-system' if M == 0 else 'undefined-at-zero-temperature-threshold' if cold else 'finite-temperature'
            tolerance = D(10) ** (-precision + 14)
            assert abs(sum(probabilities) - 1) < tolerance
            assert abs(sum(k * q for k, q in enumerate(probabilities)) - mean) < tolerance * max(abs(mean), 1)
            assert abs(sum((D(k) - mean) ** 2 * q for k, q in enumerate(probabilities)) - variance) < tolerance * max(abs(variance), 1)
            if not cold:
                identity = delta * mean - T * kb * entropy
                assert abs(identity + sign_t * theta * log_xi) < tolerance * max(abs(theta * log_xi), D('1e-9999'))
            if M <= 8:
                weights = [w ** (M - sum(bits) if positive else sum(bits)) for bits in product([0, 1], repeat=M)]
                total = sum(weights)
                for k in range(M + 1):
                    mass = sum(q for bits, q in zip(product([0, 1], repeat=M), weights) if sum(bits) == k) / total
                    assert abs(mass - probabilities[k]) < tolerance
                enumerated = True
        if delta_fraction == 0:
            # At zero bias these moments are exact rationals. In particular,
            # eps=MIN, M=2 gives Cov(E,N)=MIN/2, an exact ties-to-even zero.
            # Project the Fraction directly; a Decimal log/exp round trip can
            # land on either side of that tie at different working precisions.
            ef = Fraction.from_float(p['energyGap'])
            exact_moments = dict(meanParticles=Fraction(M, 2), meanHoles=Fraction(M, 2),
                                 numberVariance=Fraction(M, 4), meanEnergy=ef * M / 2,
                                 energyVariance=ef * ef * M / 4, energyNumberCovariance=ef * M / 4)
            if not cold:
                exact_moments['numberSusceptibility'] = Fraction(M, 4) / theta_fraction * sign_t
            for name, value in exact_moments.items():
                expected[name]['value'] = fraction_linear(value)
            for row in distribution:
                row['probability']['value'] = fraction_linear(Fraction(int(row['multiplicity']), 2 ** M))
        return dict(expected=expected, distribution=distribution, logGrandPartitionStatus=log_status, susceptibilityStatus=susceptibility_status,
                    independentlyEnumerated=enumerated, method='analytic logarithmic bounds beyond |bias|=10000' if u > 10000 else 'exact positive-side zero-temperature preparation' if cold and delta else 'Decimal count sums and analytic moments')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('scripts/fixtures/grand-occupancy-independent.json'))
    args = parser.parse_args()
    maximum, minimum = sys.float_info.max, math.ulp(0.0)
    inputs = [(2, KB, KB, 1), (2, KB, KB - KB * math.log(2), 1), (2, KB, KB + KB * math.log(2), 1),
              (4, KB, 0, 1), (8, KB * 300, KB * 100, 300), (16, KB * 600, 0, 300), (32, KB, KB, 1),
              (2, 0, 0, 1), (2, KB, 0, 0), (2, KB, 2 * KB, 0), (2, KB, KB, 0),
              (4, 0, 0, 0), (4, 0, -KB, 0), (4, 0, KB, 0), (0, 1, 2, 0), (0, KB, KB, 1),
              (2, KB * 750, 0, 1), (4, KB * 4097, 0, 1), (2, 1e-297, 0, 1e-297 / (KB * 800)),
              (2, 1e280, 0, 1e280 / (KB * 800)), (4, 1, 0, minimum), (4, maximum, -maximum, maximum),
              (4, maximum, maximum, minimum), (2, maximum, math.nextafter(maximum, 0), maximum),
              (2, 0, minimum, maximum), (2, minimum, -minimum, maximum), (2, minimum, minimum, maximum),
              (2, 0, maximum, minimum), (4, KB, 0, -1), (4, KB, 2 * KB, -1),
              (4, maximum, maximum, -minimum), (4, 1, 2, -minimum), (2, KB, KB, -1),
              (2, 1e-300, 0, 1e300), (2, 0, -minimum, minimum), (0, 0, 0, 0)]
    cases = []
    for M, epsilon, mu, T in inputs:
        p = dict(sites=M, energyGap=epsilon, chemicalPotential=mu, temperature=T)
        with localcontext() as c:
            c.prec = 80
            bias = (Fraction.from_float(mu) - Fraction.from_float(epsilon)) / (Fraction.from_float(KB) * Fraction.from_float(T)) if T else Fraction(0)
            u = abs(dec(bias))
            precision = max(220, 2 * int(-u.log10()) + 120) if u else 220
        low, high = evaluate(p, precision), evaluate(p, precision + 80)
        assert low == high, (p, precision)
        cases.append(dict(parameters=p, precisionDigits=[precision, precision + 80], **high))
    payload = dict(boltzmannConstantBinary=KB, scope='Independent finite-exclusion references; no reader integration.', cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(dict(cases=len(cases), enumerated=sum(x['independentlyEnumerated'] for x in cases), output=str(args.output))))


if __name__ == '__main__':
    main()
