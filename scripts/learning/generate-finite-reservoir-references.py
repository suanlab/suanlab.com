"""Independent exact finite-reservoir references, before runtime implementation.

Standard-library Fraction/comb plus a second Decimal precision pass. Small cases
also enumerate the actual fixed-K bit configurations, independently of comb.
"""
import argparse
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import combinations
import json
from math import comb
from pathlib import Path


def choose(n, k):
    return comb(n, k) if 0 <= k <= n else 0


def rational(value):
    return f"{value.numerator}/{value.denominator}"


def decimal(value):
    return Decimal(value.numerator) / Decimal(value.denominator)


def information(p, q, precision):
    with localcontext() as context:
        context.prec = precision
        entropy = -sum((decimal(x) * decimal(x).ln() for x in p if x), Decimal(0))
        forward = sum((decimal(x) * decimal(x / y).ln() for x, y in zip(p, q) if x and x != y), Decimal(0))
        reverse = None if any(y and not x for x, y in zip(p, q)) else sum(
            (decimal(y) * decimal(y / x).ln() for x, y in zip(p, q) if y and x != y), Decimal(0))
        assert forward >= 0 and (reverse is None or reverse >= 0)
        return [entropy, forward, reverse]


def make_case(n, b, k):
    total = n + b
    rho = Fraction(k, total)
    denominator = choose(total, k)
    counts = [choose(n, a) * choose(b, k - a) for a in range(n + 1)]
    p = [Fraction(count, denominator) for count in counts]
    q = [choose(n, a) * rho ** a * (1 - rho) ** (n - a) for a in range(n + 1)]
    assert sum(p) == sum(q) == 1
    mean = sum(a * value for a, value in enumerate(p))
    variance = sum((a - mean) ** 2 * value for a, value in enumerate(p))
    canonical_variance = n * rho * (1 - rho)
    assert mean == n * rho == sum(a * value for a, value in enumerate(q))
    assert variance == (canonical_variance * Fraction(b, total - 1) if total > 1 else 0)
    tv = sum(abs(x - y) for x, y in zip(p, q)) / 2
    enumerated = total <= 12
    if enumerated:
        observed = Counter(sum(i < n for i in excited) for excited in combinations(range(total), k))
        assert [observed[a] for a in range(n + 1)] == counts
    low, high = information(p, q, 200), information(p, q, 280)
    for x, y in zip(low, high):
        assert (x is None) == (y is None)
        if x is not None:
            assert abs(x - y) <= max(abs(y), Decimal(1)) * Decimal('1e-180')
    return {
        'subsystemSites': n, 'reservoirSites': b, 'totalExcitations': k,
        'wholeShellCount': str(denominator), 'blockCounts': [str(x) for x in counts],
        'microcanonicalMarginal': [rational(x) for x in p],
        'matchingCanonical': [rational(x) for x in q],
        'meanExcitations': rational(mean), 'varianceExcitations': rational(variance),
        'canonicalVariance': rational(canonical_variance), 'totalVariation': rational(tv),
        'entropyNats': format(high[0], '.79e'),
        'microToCanonicalKlNats': format(high[1], '.79e'),
        'canonicalToMicroKlNats': None if high[2] is None else format(high[2], '.79e'),
        'canonicalToMicroKlStatus': 'infinite' if high[2] is None else 'finite',
        'enumeratedBitConfigurations': enumerated,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('scripts/fixtures/finite-reservoir-independent.json'))
    args = parser.parse_args()
    inputs = [(0, 1, 0), (0, 4, 2), (1, 0, 0), (1, 0, 1), (1, 1, 1),
              (1, 9, 3), (2, 0, 1), (2, 2, 2), (2, 4, 2), (2, 6, 4),
              (2, 18, 10), (2, 98, 50), (3, 3, 1), (3, 3, 3), (3, 3, 5),
              (4, 8, 0), (4, 8, 1), (4, 8, 6), (4, 8, 11), (4, 8, 12),
              (8, 8, 3), (8, 8, 8), (8, 8, 13), (16, 48, 24),
              (32, 96, 48), (128, 384, 192)]
    cases = [make_case(*values) for values in inputs]
    witness = next(x for x in cases if (x['subsystemSites'], x['reservoirSites'], x['totalExcitations']) == (2, 2, 2))
    assert witness['microcanonicalMarginal'] == ['1/6', '2/3', '1/6']
    assert witness['matchingCanonical'] == ['1/4', '1/2', '1/4']
    assert witness['totalVariation'] == '1/6' and witness['varianceExcitations'] == '1/3'
    payload = {
        'scope': 'Independent reference preparation; runtime model, manuscripts and reader integration remain unauthored.',
        'method': 'Exact integer configuration counts and Fraction probabilities; Decimal information at 200 and 280 digits, agreement checked at 180 decimal places; tiny systems independently enumerated.',
        'cases': cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'cases': len(cases), 'enumeratedCases': sum(x['enumeratedBitConfigurations'] for x in cases), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
