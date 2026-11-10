/** Full independent reconstruction; untrusted response strings are never
 * parsed as arithmetic. Only the bounded original input supplies operands.
 */
import { readQuarticAsymptoticInput, type QuarticAsymptoticInput } from './learning-quartic-asymptotic-input';
import { quarticAsymptoticContract } from './learning-quartic-asymptotic-contract';
import { independentlyEncloseQuarticIntegral } from './learning-quartic-asymptotic-validation-integral';
import { Rational as R, Z, O } from './learning-graph-gauge-validation-math';
import { snapshotQubitWire } from './learning-qubit-wire';
import type { QuarticAsymptoticResult } from './learning-quartic-asymptotic-protocol';
const B = BigInt;
const compare = (a: R, b: R) => a.n * b.d < b.n * a.d ? -1 : a.n * b.d > b.n * a.d ? 1 : 0;
const abs = (a: R) => a.n < B(0) ? a.neg() : a;
const min = (a: R, b: R) => compare(a, b) <= 0 ? a : b, max = (a: R, b: R) => compare(a, b) >= 0 ? a : b;
const factorial = (n: number) => { let v = B(1); for (let i = 2; i <= n; i++) v *= B(i); return v; };
function power(x: R, n: number) { let value = O; for (let i = 0; i < n; i++) value = value.mul(x); return value; }
function isqrt(n: bigint) {
  if (n < B(2)) return n;
  let x = B(1) << B(Math.ceil(n.toString(2).length / 2));
  for (;;) { const next = (x + n / x) / B(2); if (next >= x) return x; x = next; }
}
function globalOrder(g: R) {
  if (g.n === B(0)) return 0;
  const a = B(16) * g.n, b = B(48) * g.n - B(4) * g.d, c = B(35) * g.n - B(8) * g.d;
  if (c >= B(0)) return 0;
  let n = (-b + isqrt(b * b - B(4) * a * c)) / (B(2) * a);
  const evaluate = (x: bigint) => a * x * x + b * x + c;
  while (evaluate(n) < B(0)) n++;
  while (n > B(0) && evaluate(n - B(1)) >= B(0)) n--;
  return Number(n);
}
function same(value: unknown, expected: unknown): boolean {
  if (expected === null || typeof expected !== 'object') return value === expected;
  if (!value || typeof value !== 'object' || Array.isArray(value) !== Array.isArray(expected)) return false;
  if (Array.isArray(expected)) return Array.isArray(value) && value.length === expected.length && expected.every((item, i) => same(value[i], item));
  const left = value as Record<string, unknown>, right = expected as Record<string, unknown>, keys = Object.keys(right);
  return Object.keys(left).length === keys.length && keys.every(key => Object.hasOwn(left, key) && same(left[key], right[key]));
}
type Integral = ReturnType<typeof independentlyEncloseQuarticIntegral>;
let gaussian: Integral | undefined;
const cache = new Map<string, Integral>();
function integralFor(g: R) {
  const key = g.text(), existing = cache.get(key); if (existing) return existing;
  const computed = independentlyEncloseQuarticIntegral(g);
  if (cache.size === 16) cache.delete(cache.keys().next().value!);
  cache.set(key, computed); return computed;
}
export function validateQuarticAsymptoticResult(source: QuarticAsymptoticInput, value: unknown): QuarticAsymptoticResult | null {
  try {
    const input = readQuarticAsymptoticInput(source); if (!input) return null;
    const owned = snapshotQubitWire(value), g = R.read(input.coupling); gaussian ??= integralFor(Z);
    const integral = g.n === B(0) ? gaussian : integralFor(g);
    const lo = g.n === B(0) ? O : integral.lower.div(gaussian.upper), hi = g.n === B(0) ? O : integral.upper.div(gaussian.lower);
    const coefficients = Array.from({ length: input.order + 2 }, (_, n) => new R((n % 2 ? -B(1) : B(1)) * factorial(4 * n), (B(1) << B(4 * n)) * factorial(2 * n) * factorial(n)));
    const terms = coefficients.map((c, n) => c.mul(power(g, n)));
    const partials = terms.map((_, n) => terms.slice(0, n + 1).reduce((sum, term) => sum.add(term), Z));
    const lower = partials.filter((_, n) => n % 2).reduce(max, Z), upper = partials.filter((_, n) => !(n % 2)).reduce(min, O);
    const approximations = partials.slice(0, input.order + 1).map((p, order) => {
      const omitted = abs(terms[order + 1]), leftDistance = abs(p.sub(lo)), rightDistance = abs(p.sub(hi));
      const distanceLower = compare(lo, p) <= 0 && compare(p, hi) <= 0 ? Z : min(leftDistance, rightDistance), distanceUpper = max(leftDistance, rightDistance);
      return { order, coefficient: coefficients[order].text(), term: terms[order].text(), partial: p.text(), firstOmittedMagnitude: omitted.text(),
        position: g.n === B(0) ? 'exact' : order % 2 ? 'below' : 'above',
        quadratureDistance: { lower: distanceLower.text(), upper: distanceUpper.text() }, errorInterval: { lower: distanceLower.text(), upper: min(distanceUpper, omitted).text() } };
    });
    const best = terms.slice(1).reduce((index, term, j) => compare(abs(term), abs(terms[index + 1])) < 0 ? j : index, 0), global = globalOrder(g);
    const expected = { input, series: g.n === B(0) ? 'exact-gaussian' : 'divergent-asymptotic', coefficients: coefficients.map(c => c.text()), approximations,
      quadratureInterval: { lower: lo.text(), upper: hi.text(), width: hi.sub(lo).text() }, asymptoticEnvelope: { lower: lower.text(), upper: upper.text() }, combinedInterval: { lower: max(lo, lower).text(), upper: min(hi, upper).text() },
      bestBoundOrderWithinBudget: best, globallyBestBoundOrder: global, budgetReachesBestBound: global <= input.order,
      quadrature: integral.certificate, gaussianNormalization: gaussian.certificate, contract: quarticAsymptoticContract };
    return same(owned, expected) ? owned as QuarticAsymptoticResult : null;
  } catch { return null; }
}
