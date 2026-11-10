/** Certified half-line quadrature, independent of the perturbative coefficients.
 * exp(-S(left)) times the alternating Taylor polynomials of exp(-ΔS(u)).
 * ΔS has nonnegative coefficients on u∈[0,1]. All rounding is directed.
 */
import { createTranscendentalContext } from './learning-interval-transcendentals';
import { f, zero, one, add, mul, div, type F } from './learning-quantum-hilbert-core';
const B = BigInt, precisionBits = 192, cells = 256, cutoff = 16, degree = 32;
const scale = B(1) << B(precisionBits), ctx = createTranscendentalContext(precisionBits);
type I = { lower: bigint; upper: bigint };
const ceil = (n: bigint, d: bigint) => (n + d - B(1)) / d;
const interval = (x: F): I => ({ lower: x.n * scale / x.d, upper: ceil(x.n * scale, x.d) });
const times = (a: I, b: I): I => ({ lower: a.lower * b.lower / scale, upper: ceil(a.upper * b.upper, scale) });
const integer = (n: number) => f(B(n));
const pow = (x: F, n: number) => { let v = one; for (let i = 0; i < n; i++) v = mul(v, x); return v; };
const exp = (x: F): I => {
  if (x.n === B(0)) return { lower: scale, upper: scale };
  const numerator = { coefficient: x.n, exponent: 0 }, denominator = { coefficient: x.d, exponent: 0 };
  const initial = ctx.expNegativeRatio(numerator, denominator);
  if (initial.upper - initial.lower === B(1)) return initial;
  // Positive rational exponents have irrational exponentials. Resolve one
  // target cell before selecting the canonical dyadic floor and ceiling.
  for (let bits = precisionBits + 64; bits <= 2048; bits += 64) {
    const refined = createTranscendentalContext(bits).expNegativeRatio(numerator, denominator);
    const shift = B(bits - precisionBits), lo = refined.lower >> shift, hi = refined.upper >> shift;
    if (lo === hi) return { lower: lo, upper: lo + B(1) };
  }
  throw new RangeError('Exponential target cell unresolved');
};
const text = (x: F) => x.d === B(1) ? String(x.n) : `${x.n}/${x.d}`;
export function encloseQuarticHalfIntegral(g: F) {
  const width = f(B(cutoff), B(cells)); let totalLower = B(0), totalUpper = B(0), skippedTinyPanels = 0;
  for (let panel = 0; panel < cells; panel++) {
    const x = mul(integer(panel), width), x2 = pow(x, 2), action = add(div(x2, integer(2)), div(mul(g, pow(x, 4)), integer(4)));
    const factor = exp(action);
    if (factor.upper <= B(1)) { totalUpper += factor.upper; skippedTinyPanels++; continue; }
    const delta = [zero,
      mul(add(x, mul(g, pow(x, 3))), width),
      mul(add(f(B(1), B(2)), mul(f(B(3), B(2)), mul(g, x2))), pow(width, 2)),
      mul(mul(g, x), pow(width, 3)), div(mul(g, pow(width, 4)), integer(4)),
    ].map(interval);
    let polynomial: I[] = [{ lower: scale, upper: scale }], sum: I = { lower: scale, upper: scale }, evenUpper = scale, factorial = B(1);
    for (let n = 1; n <= degree + 1; n++) {
      const next: I[] = Array.from({ length: polynomial.length + 4 }, () => ({ lower: B(0), upper: B(0) }));
      for (let i = 0; i < polynomial.length; i++) for (let j = 1; j <= 4; j++) {
        const p = times(polynomial[i], delta[j]); next[i + j].lower += p.lower; next[i + j].upper += p.upper;
      }
      polynomial = next; factorial *= B(n); let lower = B(0), upper = B(0);
      for (let i = 0; i < polynomial.length; i++) { const denominator = B(i + 1) * factorial; lower += polynomial[i].lower / denominator; upper += ceil(polynomial[i].upper, denominator); }
      sum = n % 2 ? { lower: sum.lower - upper, upper: sum.upper - lower } : { lower: sum.lower + lower, upper: sum.upper + upper };
      if (n === degree) evenUpper = sum.upper;
    }
    if (evenUpper < B(0)) throw Error('Invalid even Taylor enclosure');
    const lower = factor.lower * (sum.lower > B(0) ? sum.lower : B(0)) / scale;
    const rawUpper = ceil(factor.upper * evenUpper, scale), upper = rawUpper < factor.upper ? rawUpper : factor.upper;
    if (lower > upper) throw Error('Empty quadrature panel');
    totalLower += lower; totalUpper += upper;
  }
  const L = integer(cutoff), tailAction = add(div(pow(L, 2), integer(2)), div(mul(g, pow(L, 4)), integer(4)));
  const tailUpper = div(f(exp(tailAction).upper, scale), add(L, mul(g, pow(L, 3))));
  const lower = mul(f(totalLower, scale), width), upper = add(mul(f(totalUpper, scale), width), tailUpper);
  return { lower, upper, certificate: { cutoff, cells, precisionBits, localTaylorEvenDegree: degree, localTaylorOddDegree: degree + 1,
    omittedTinyPanels: skippedTinyPanels, halfLineLower: text(lower), halfLineUpper: text(upper), tailUpper: text(tailUpper) } };
}
