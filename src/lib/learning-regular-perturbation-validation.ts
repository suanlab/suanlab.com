/** Independent inverse-series coefficients and integer-grid root enclosure.
 * No producer engine or producer rational/polynomial helper is imported.
 * Output strings are compared as text, never used as arithmetic operands.
 */
import { readRegularPerturbationInput, type RegularPerturbationInput } from './learning-regular-perturbation-input';
import { Rational as R, Z, O, T } from './learning-graph-gauge-validation-math';
import { snapshotQubitWire } from './learning-qubit-wire';
import type { RegularPerturbationResult } from './learning-regular-perturbation-protocol';

const integer = (n: number) => new R(BigInt(n));
const cmp = (a: R, b: R) => a.n * b.d < b.n * a.d ? -1 : a.n * b.d > b.n * a.d ? 1 : 0;
const abs = (a: R) => a.n < BigInt(0) ? a.neg() : a;
const minimum = (a: R, b: R) => cmp(a, b) <= 0 ? a : b;
const maximum = (a: R, b: R) => cmp(a, b) >= 0 ? a : b;
function power(base: R, exponent: number) {
  let out = O, factor = base;
  for (let n = exponent; n; n = Math.floor(n / 2)) { if (n % 2) out = out.mul(factor); factor = factor.mul(factor); }
  return out;
}
const factorial = (n: number) => { let out = BigInt(1); for (let i = 2; i <= n; i++) out *= BigInt(i); return out; };
function same(value: unknown, expected: unknown): boolean {
  if (expected === null || typeof expected !== 'object') return value === expected;
  if (!value || typeof value !== 'object' || Array.isArray(value) !== Array.isArray(expected)) return false;
  if (Array.isArray(expected)) return Array.isArray(value) && value.length === expected.length && expected.every((item, i) => same(value[i], item));
  const left = value as Record<string, unknown>, right = expected as Record<string, unknown>, keys = Object.keys(right);
  return Object.keys(left).length === keys.length && keys.every(key => Object.hasOwn(left, key) && same(left[key], right[key]));
}

export function validateRegularPerturbationResult(source: RegularPerturbationInput, value: unknown): RegularPerturbationResult | null {
  try {
    const input = readRegularPerturbationInput(source);
    if (!input) return null;
    const owned = snapshotQubitWire(value);
    const k = R.read(input.stiffness), lambda = R.read(input.coupling), h = R.read(input.source);
    const F = (x: R) => k.mul(x).add(lambda.mul(power(x, 3))).sub(h);
    const scale = h.div(k), g = lambda.mul(power(h, 2)).div(power(k, 3));
    const counts = Array.from({ length: input.order + 1 }, (_, n) =>
      (n % 2 ? -BigInt(1) : BigInt(1)) * factorial(3 * n) / (factorial(n) * factorial(2 * n + 1)));
    const coefficients = counts.map((count, n) => {
      // Lagrange inversion: (3n)!/[n!(2n+1)!], independent of coefficient matching.
      return new R(count).mul(power(h, 2 * n + 1)).div(power(k, 3 * n + 1));
    });
    const origin = minimum(Z, scale), end = maximum(Z, scale), span = end.sub(origin);
    const denominator = BigInt(1) << BigInt(48);
    const point = (index: bigint) => origin.add(span.mul(new R(index, denominator)));
    let left = BigInt(0), right = denominator;
    if (F(origin).n === BigInt(0)) right = left;
    else if (F(end).n === BigInt(0)) left = right;
    else while (right - left > BigInt(1)) {
      const index = (left + right) / BigInt(2), sign = F(point(index)).n;
      if (sign === BigInt(0)) { left = index; right = index; }
      else if (sign < BigInt(0)) left = index;
      else right = index;
    }
    const lower = point(left), upper = point(right);
    const approximations = coefficients.map((coefficient, order) => {
      const approximation = coefficients.slice(0, order + 1).reduce((total, c, i) => total.add(c.mul(power(lambda, i))), Z);
      const residual = F(approximation), a = abs(approximation.sub(lower)), b = abs(approximation.sub(upper));
      const inside = cmp(lower, approximation) <= 0 && cmp(approximation, upper) <= 0;
      return { order, coefficient: coefficient.text(), term: coefficient.mul(power(lambda, order)).text(), approximation: approximation.text(), residual: residual.text(),
        errorBound: abs(residual).div(k).text(), energyGapBound: power(residual, 2).div(T.mul(k)).text(),
        rootDistance: { lower: (inside ? Z : minimum(a, b)).text(), upper: maximum(a, b).text() } };
    });
    // Expand unordered triples with their multinomial multiplicities, rather
    // than call the producer's two polynomial convolutions.
    const residualPolynomial = Array.from({ length: 3 * input.order + 2 }, (_, i) => (coefficients[i] ?? Z).mul(k).sub(i === 0 ? h : Z));
    const tripleCounts = Array<bigint>(3 * input.order + 1).fill(BigInt(0));
    for (let i = 0; i <= input.order; i++) for (let j = i; j <= input.order; j++) for (let l = j; l <= input.order; l++) {
      const multiplicity = i === l ? 1 : i === j || j === l ? 3 : 6;
      tripleCounts[i + j + l] += counts[i] * counts[j] * counts[l] * BigInt(multiplicity);
    }
    // Every triple of total degree s has the common factor h^(2s+3)/k^(3s+3).
    // Sum integer multiplicities first; avoid thousands of huge rational GCDs.
    tripleCounts.forEach((count, s) => {
      residualPolynomial[s + 1] = residualPolynomial[s + 1].add(new R(count).mul(power(h, 2 * s + 3)).div(power(k, 3 * s + 3)));
    });
    const lowSquare = power(lower, 2), highSquare = power(upper, 2);
    const minSquare = cmp(lower, Z) <= 0 && cmp(Z, upper) <= 0 ? Z : minimum(lowSquare, highSquare);
    const maxSquare = maximum(lowSquare, highSquare), threeLambda = integer(3).mul(lambda);
    const critical = cmp(integer(27).mul(g), integer(4));
    const expected = { input, dimensionlessCoupling: g.text(), linearScale: scale.text(),
      convergenceRegion: h.n === BigInt(0) ? 'identically-zero' : critical === 0 ? 'radius-boundary' : critical < 0 ? 'inside-radius' : 'outside-radius',
      coefficients: coefficients.map(c => c.text()), approximations, residualPolynomial: residualPolynomial.map(c => c.text()), cancelledThroughOrder: input.order,
      evaluatedResidual: approximations.at(-1)!.residual,
      root: { lower: lower.text(), upper: upper.text(), width: upper.sub(lower).text(), lowerResidual: F(lower).text(), upperResidual: F(upper).text(), bisectionLimit: 48 },
      susceptibility: { lower: O.div(k.add(threeLambda.mul(maxSquare))).text(), upper: O.div(k.add(threeLambda.mul(minSquare))).text() } };
    return same(owned, expected) ? owned as RegularPerturbationResult : null;
  } catch { return null; }
}
