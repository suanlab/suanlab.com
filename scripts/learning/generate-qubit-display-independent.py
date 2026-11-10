"""Independent rational/isqrt certificates, including severe cancellation.
Run from the project root; only rewrites its own fixture.
"""
import json
from math import isqrt
from pathlib import Path
import sympy as s

SCALE = 2**40
cases = []


def append(name, a, b=0, limit=1):
    a, b = s.Rational(a), s.Rational(b)
    if b == 0:
        sign = int(s.sign(a))
        outside = abs(a) > limit
        low = int(s.floor(a * SCALE))
        exact = a == s.Rational(low, SCALE)
    else:
        # Independent route: tighten rational sqrt(2) brackets, rather than
        # the product's exact sign comparisons and binary search on x.
        bits = 64
        while True:
            denominator = 2**bits
            root = isqrt(2 * denominator**2)
            endpoints = [a + b * s.Rational(root, denominator),
                         a + b * s.Rational(root + 1, denominator)]
            left, right = min(endpoints), max(endpoints)
            sign = 1 if left >= 0 else -1 if right <= 0 else None
            outside = True if left >= limit or right <= -limit else False if left >= -limit and right <= limit else None
            low = int(s.floor(left * SCALE))
            if sign is not None and outside is not None and (outside or low == int(s.floor(right * SCALE))):
                break
            bits *= 2
            if bits > 16384:
                raise ArithmeticError("Independent bracket did not converge")
        exact = False  # Nonzero rational b makes a+b*sqrt(2) irrational.
    if outside:
        expected = dict(sign=sign, status="outside", lower=None, upper=None,
                        coordinate=None, exact=False)
    else:
        lower, upper = low / SCALE, (low if exact else low + 1) / SCALE
        expected = dict(sign=sign, status="enclosed", lower=lower, upper=upper,
                        coordinate=(lower + upper) / 2, exact=exact)
    cases.append(dict(name=name, coefficients=[str(a), str(b)], limit=limit, expected=expected))


for n in range(-12, 13):
    append(f"rational-{n}", s.Rational(n, 13))
    append(f"quadratic-{n}", s.Rational(n, 17), s.Rational(1, 7))
for limit in (1, 2, 4, 16, 1024):
    for sign in (-1, 1):
        append(f"boundary-{limit}-{sign}", sign * limit, limit=limit)
        append(f"outside-{limit}-{sign}", sign * (limit + s.Rational(1, 10**400)), limit=limit)
for sign in (-1, 1):
    append(f"tiny-{sign}", s.Rational(sign, 10**500))
    append(f"tiny-irrational-{sign}", 0, s.Rational(sign, 10**500))
    append(f"huge-coefficients-{sign}", sign * 10**500, 1)
# Pell convergents: individual coefficients exceed Number's finite range, but
# their difference is a nonzero number smaller than the display resolution.
p, q = 1, 0
for step in range(1, 1201):
    p, q = p + 2*q, p + q
    if step in (40, 400, 1200):
        append(f"pell-{step}", p, -q)
        append(f"pell-negative-{step}", -p, q)

destination = Path("scripts/fixtures/qubit-display-independent.json")
destination.write_text(json.dumps(dict(schemaVersion=1, bits=40, cases=cases), indent=2) + "\n")
print(json.dumps(dict(cases=len(cases), fixture=str(destination))))
