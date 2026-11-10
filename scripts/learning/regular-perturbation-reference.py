"""Independent Fraction/binomial and SymPy polynomial references; no TS execution."""
import json
import math
import random
from fractions import Fraction as Q
from pathlib import Path

import sympy as sp


def q(value):
    value = Q(value)
    return str(value.numerator) if value.denominator == 1 else str(value)


examples = [
    ("weak-source", "1", "1/10", "1", 4),
    ("linear-limit", "2", "0", "3", 6),
    ("zero-source", "1", "16", "0", 16),
    ("negative-source", "1", "1/10", "-1", 4),
    ("small-coupling-large-source", "1", "1/100", "4", 10),
    ("soft-stiffness", "1/10", "1/1000", "1", 8),
    ("stiff-response", "4", "1", "1", 8),
    ("critical-radius", "3", "4", "1", 12),
    ("outside-radius", "1", "1", "1", 10),
    ("exact-rational-root", "1", "1", "2", 4),
    ("zeroth-order", "1", "1/10", "1", 0),
    ("equivalent-scale", "2", "1/5", "2", 4),
    ("fractional-coefficients", "3/2", "2/7", "-5/3", 7),
    ("maximum-soft", "1/1000", "16", "16", 16),
]
random_source = random.Random(20261110)
for i in range(36):
    examples.append((f"seeded-{i}", q(Q(random_source.randint(1, 16), random_source.randint(1, 12))),
                     q(Q(random_source.randint(0, 16), random_source.randint(1, 20))),
                     q(Q(random_source.randint(-16, 16), random_source.randint(1, 12))),
                     random_source.randint(0, 16)))
examples += [("maximum-fraction-spelling", "999999999999/999999999999", "999999999999/999999999999", "-999999999999/999999999999", 16),
             ("tiny-coupling", "1", "1/999999999999", "16", 16)]

z, x = sp.symbols("z x")
rows = []
for identifier, ks, ls, hs, order in examples:
    k, lam, h = map(Q, (ks, ls, hs))
    # Lagrange inversion formula: independent of triple-convolution matching.
    coefficients = [Q((-1) ** n * math.comb(3 * n, n), 2 * n + 1) * h ** (2 * n + 1) / k ** (3 * n + 1)
                    for n in range(order + 1)]
    polynomial = sum(sp.Rational(c.numerator, c.denominator) * z ** n for n, c in enumerate(coefficients))
    residual = sp.Poly(k * polynomial + z * polynomial ** 3 - h, z)
    residual_coefficients = [q(residual.nth(i)) for i in range(3 * order + 2)]
    root_polynomial = sp.Poly(lam * x ** 3 + k * x - h, x)
    intervals = sp.polys.polytools.intervals(root_polynomial, eps=sp.Rational(1, 2 ** 60))
    assert len(intervals) == 1 and intervals[0][1] == 1
    lower, upper = intervals[0][0]
    partial = Q(0)
    approximations = []
    for n, coefficient in enumerate(coefficients):
        term = coefficient * lam ** n
        partial += term
        r = k * partial + lam * partial ** 3 - h
        approximations.append({"order": n, "coefficient": q(coefficient), "term": q(term),
                               "approximation": q(partial), "residual": q(r),
                               "errorBound": q(abs(r) / k), "energyGapBound": q(r * r / (2 * k))})
    rows.append({"id": identifier, "input": {"stiffness": ks, "coupling": ls, "source": hs, "order": order},
                 "coefficients": list(map(q, coefficients)), "residualPolynomial": residual_coefficients,
                 "approximations": approximations, "dimensionlessCoupling": q(lam * h * h / k ** 3),
                 "independentRoot": {"lower": q(lower), "upper": q(upper)}})
destination = Path("scripts/fixtures/regular-perturbation-reference.json")
destination.write_text(json.dumps({"method": "Lagrange inversion plus SymPy polynomial expansion and root isolation", "cases": rows}, indent=2) + "\n")
print(f"{len(rows)} independent reference cases -> {destination}")
