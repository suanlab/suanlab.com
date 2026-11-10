"""Independent Gaussian moments, mpmath quadrature and Bessel-K cross-check.
The decimal reference is high-precision numerical evidence, not an interval proof.
"""
import json
import math
import random
from fractions import Fraction as Q
from pathlib import Path

import mpmath as mp


def q(x):
    x = Q(x)
    return str(x.numerator) if x.denominator == 1 else str(x)


examples = [
    ("useful-truncation", "1/10", 12), ("gaussian-exact", "0", 12),
    ("smaller-coupling", "1/100", 32), ("budget-before-minimum", "1/1000", 32),
    ("strong-coupling", "1", 12), ("maximum-coupling", "16", 32),
    ("zero-order", "1/10", 0), ("first-order", "1/10", 1),
    ("least-bound-order", "1/10", 2), ("beyond-useful-order", "1/10", 32),
    ("equal-omitted-terms", "8/35", 8), ("equivalent-spelling", "2/20", 12),
    ("tiny-positive", "1/999999999989", 32),
    ("maximum-coprime", "999999999983/62499999999", 32),
]
rng = random.Random(20261110)
for i in range(24):
    examples.append((f"seeded-{i}", q(Q(rng.randrange(1, 160), rng.randrange(10, 1000))), rng.randrange(33)))

mp.mp.dps = 140
rows = []
for identifier, raw_g, order in examples:
    g = Q(raw_g)
    coefficients = [Q((-1) ** n * math.factorial(4 * n), 2 ** (4 * n) * math.factorial(2 * n) * math.factorial(n)) for n in range(order + 2)]
    terms = [c * g ** n for n, c in enumerate(coefficients)]
    partial = Q(0)
    approximations = []
    for n in range(order + 1):
        partial += terms[n]
        approximations.append({"order": n, "coefficient": q(coefficients[n]), "term": q(terms[n]), "partial": q(partial),
                               "firstOmittedMagnitude": q(abs(terms[n + 1])), "position": "exact" if not g else "below" if n % 2 else "above"})
    best = min(range(order + 1), key=lambda n: abs(terms[n + 1]))
    if not g:
        global_best = 0
        quadrature = bessel = mp.mpf(1)
    else:
        a, b, c = 16 * g.numerator, 48 * g.numerator - 4 * g.denominator, 35 * g.numerator - 8 * g.denominator
        if c >= 0:
            global_best = 0
        else:
            global_best = max(0, (-b + math.isqrt(b * b - 4 * a * c)) // (2 * a))
            polynomial = lambda n: a * n * n + b * n + c
            while polynomial(global_best) < 0:
                global_best += 1
            while global_best and polynomial(global_best - 1) >= 0:
                global_best -= 1
        gm = mp.mpf(g.numerator) / g.denominator
        quadrature = 2 * mp.quad(lambda x: mp.exp(-x * x / 2 - gm * x ** 4 / 4), [0, 1, 2, 4, 8, mp.inf]) / mp.sqrt(2 * mp.pi)
        z = 1 / (8 * gm)
        bessel = mp.exp(z) * mp.besselk(mp.mpf(1) / 4, z) / (2 * mp.sqrt(mp.pi * gm))
        assert abs(quadrature - bessel) < mp.mpf("1e-115"), identifier
    rows.append({"id": identifier, "input": {"coupling": raw_g, "order": order}, "coefficients": list(map(q, coefficients)),
                 "approximations": approximations, "bestBoundOrderWithinBudget": best, "globallyBestBoundOrder": global_best,
                 "referenceDecimal": mp.nstr(quadrature, 110, strip_zeros=False, min_fixed=-200, max_fixed=200),
                 "referenceMethodsDifference": mp.nstr(abs(quadrature - bessel), 8),
                 "referenceIsCertifiedInterval": False})
destination = Path("scripts/fixtures/quartic-asymptotic-reference.json")
destination.write_text(json.dumps({"method": "Factorial Gaussian moments; mpmath 140-digit quadrature and Bessel-K expression; integer discriminant order selection", "cases": rows}, indent=2) + "\n")
print(f"{len(rows)} independent reference cases -> {destination}")
