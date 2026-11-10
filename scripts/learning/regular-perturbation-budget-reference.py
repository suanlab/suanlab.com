"""Independent exact checks for coprime input fractions at the declared limits."""
import hashlib
import json
import math
import sys
from fractions import Fraction as Q
from pathlib import Path

import sympy as sp

sys.set_int_max_str_digits(30000)


def q(x):
    x = Q(x)
    return str(x.numerator) if x.denominator == 1 else str(x)


z = sp.symbols("z")
cases = json.loads(Path("scripts/fixtures/regular-perturbation-budget.json").read_text())
rows = []
for case in cases:
    data = case["input"]
    k, lam, h = (Q(data[key]) for key in ["stiffness", "coupling", "source"])
    count = data["order"]
    coefficients = [Q((-1) ** n * math.comb(3 * n, n), 2 * n + 1) * h ** (2 * n + 1) / k ** (3 * n + 1)
                    for n in range(count + 1)]
    polynomial = sum(sp.Rational(c.numerator, c.denominator) * z ** n for n, c in enumerate(coefficients))
    residual = sp.Poly(k * polynomial + z * polynomial ** 3 - h, z)
    partial = Q(0)
    approximations = []
    for n, c in enumerate(coefficients):
        term = c * lam ** n
        partial += term
        r = k * partial + lam * partial ** 3 - h
        approximations.append({"order": n, "coefficient": q(c), "term": q(term), "approximation": q(partial),
                               "residual": q(r), "errorBound": q(abs(r) / k), "energyGapBound": q(r * r / (2 * k))})
    payload = {"coefficients": list(map(q, coefficients)), "residualPolynomial": [q(residual.nth(i)) for i in range(3 * count + 2)],
               "dimensionlessCoupling": q(lam * h * h / k ** 3), "approximations": approximations}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    rows.append({"id": case["id"], "input": data, "payloadSha256": digest, "method": "Fraction closed binomial coefficients and SymPy symbolic residual"})
destination = Path("scripts/fixtures/regular-perturbation-budget-reference.json")
destination.write_text(json.dumps(rows, indent=2) + "\n")
print(f"{len(rows)} maximum-input independent payload hashes -> {destination}")
