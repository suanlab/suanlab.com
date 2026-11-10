/** Independent rational interval arithmetic for Schmidt attribution.
 * Every operation rounds outwards to a 3072-bit dyadic grid. No production
 * Ritz, root, Hilbert or transcendental engine is imported.
 */
import { zero, one, neg, le, c, type F, type Range, type C, type Matrix } from './learning-tensor-wire-math';
export type WireIntervalComplex = { re: Range; im: Range };
export type WireIntervalMatrix = WireIntervalComplex[][];
const bits = 3072, scale = BigInt(1) << BigInt(bits);
const floor = (n: bigint, d: bigint): bigint => d < BigInt(0) ? floor(-n, -d) : n >= BigInt(0) ? n / d : -((-n + d - BigInt(1)) / d), ceil = (n: bigint, d: bigint) => -floor(-n, d);
const minimum = (xs: bigint[]) => xs.reduce((a, b) => a < b ? a : b), maximum = (xs: bigint[]) => xs.reduce((a, b) => a > b ? a : b);
// F permits an unreduced positive denominator. Keeping a common dyadic grid
// avoids thousands of gcds in every small matrix product.
const grid = (lower: bigint, upper = lower): Range => ({ lower: { n: lower, d: scale }, upper: { n: upper, d: scale } });
const low = (a: Range) => a.lower.d === scale ? a.lower.n : floor(a.lower.n * scale, a.lower.d);
const high = (a: Range) => a.upper.d === scale ? a.upper.n : ceil(a.upper.n * scale, a.upper.d);
export const wireRange = (lower: F, upper = lower): Range => {
  if (!le(lower, upper)) throw new RangeError('Ordered independent interval required');
  return grid(floor(lower.n * scale, lower.d), ceil(upper.n * scale, upper.d));
};
export const wireAdd = (a: Range, b: Range) => grid(low(a) + low(b), high(a) + high(b));
export const wireNeg = (a: Range) => grid(-high(a), -low(a));
export const wireSub = (a: Range, b: Range) => grid(low(a) - high(b), high(a) - low(b));
export const wireMul = (a: Range, b: Range) => {
  const al = low(a), au = high(a), bl = low(b), bu = high(b), values = [al * bl, al * bu, au * bl, au * bu];
  return grid(floor(minimum(values), scale), ceil(maximum(values), scale));
};
export const wireDiv = (a: Range, b: Range) => {
  const al = low(a), au = high(a), bl = low(b), bu = high(b);
  if (bl <= BigInt(0) && bu >= BigInt(0)) throw new RangeError('Independent divisor interval contains zero');
  const pairs = [[al, bl], [al, bu], [au, bl], [au, bu]];
  return grid(minimum(pairs.map(([n, d]) => floor(n * scale, d))), maximum(pairs.map(([n, d]) => ceil(n * scale, d))));
};
export const wireSquare = (a: Range) => {
  const l = low(a), u = high(a), values = [l * l, u * u];
  return grid(l <= BigInt(0) && u >= BigInt(0) ? BigInt(0) : floor(minimum(values), scale), ceil(maximum(values), scale));
};
function integerSquareRoot(n: bigint): bigint {
  if (n < BigInt(0)) throw new RangeError('Nonnegative square required'); if (n < BigInt(2)) return n;
  let a = BigInt(1) << BigInt(Math.ceil(n.toString(2).length / 2));
  for (;;) { const b = (a + n / a) / BigInt(2); if (b >= a) return a; a = b; }
}
export const wireSqrt = (a: Range): Range => {
  const lower = low(a), upper = high(a);
  if (upper < BigInt(0)) throw new RangeError('Independent nonnegative square-root interval required');
  const l = integerSquareRoot((lower < BigInt(0) ? BigInt(0) : lower) * scale), u = integerSquareRoot(upper * scale);
  return grid(l, u + (u * u === upper * scale ? BigInt(0) : BigInt(1)));
};
export const wireComplex = (v: C): WireIntervalComplex => ({ re: wireRange(v.re), im: wireRange(v.im) });
export const wireComplexAdd = (a: WireIntervalComplex, b: WireIntervalComplex): WireIntervalComplex => ({ re: wireAdd(a.re, b.re), im: wireAdd(a.im, b.im) });
export const wireComplexScale = (a: WireIntervalComplex, b: Range): WireIntervalComplex => ({ re: wireMul(a.re, b), im: wireMul(a.im, b) });
export const wireComplexMul = (a: WireIntervalComplex, b: WireIntervalComplex): WireIntervalComplex => ({ re: wireSub(wireMul(a.re, b.re), wireMul(a.im, b.im)), im: wireAdd(wireMul(a.re, b.im), wireMul(a.im, b.re)) });
export const wireMatrix = (a: Matrix): WireIntervalMatrix => a.map(row => row.map(wireComplex));
export const wireZeros = (n: number, m = n): WireIntervalMatrix => Array.from({ length: n }, () => Array.from({ length: m }, () => wireComplex(c(zero))));
export const wireIdentity = (n: number) => wireZeros(n).map((row, i) => row.map((v, j) => i === j ? wireComplex(c(one)) : v));
export const wireMatrixAdd = (a: WireIntervalMatrix, b: WireIntervalMatrix) => a.map((row, i) => row.map((v, j) => wireComplexAdd(v, b[i][j])));
export const wireMatrixScale = (a: WireIntervalMatrix, b: Range) => a.map(row => row.map(v => wireComplexScale(v, b)));
export const wireMatrixSub = (a: WireIntervalMatrix, b: WireIntervalMatrix) => wireMatrixAdd(a, wireMatrixScale(b, wireRange(neg(one))));
export const wireProduct = (a: WireIntervalMatrix, b: WireIntervalMatrix) => a.map(row => b[0].map((_, j) => row.reduce((s, v, k) => wireComplexAdd(s, wireComplexMul(v, b[k][j])), wireComplex(c(zero)))));
export const wireMatrixNormSquared = (a: WireIntervalMatrix) => a.flat().reduce((s, v) => wireAdd(s, wireAdd(wireSquare(v.re), wireSquare(v.im))), wireRange(zero));
/** Evaluate the Lagrange interpolation polynomial by matrix Horner steps,
 * rather than the production chain of matrix factors and gap inverses.
 */
export function wireSpectralProjector(density: Matrix, roots: readonly Range[], index: number): WireIntervalMatrix {
  let coefficients = [wireRange(one)], denominator = wireRange(one);
  roots.forEach((r, j) => {
    if (j === index) return;
    const next = Array.from({ length: coefficients.length + 1 }, () => wireRange(zero));
    coefficients.forEach((a, i) => { next[i] = wireSub(next[i], wireMul(a, r)); next[i + 1] = wireAdd(next[i + 1], a); });
    coefficients = next; denominator = wireMul(denominator, wireSub(roots[index], r));
  });
  const I = wireIdentity(density.length), H = wireMatrix(density);
  let value = wireMatrixScale(I, coefficients.at(-1)!);
  for (let i = coefficients.length - 2; i >= 0; i--) value = wireMatrixAdd(wireProduct(value, H), wireMatrixScale(I, coefficients[i]));
  return wireMatrixScale(value, wireDiv(wireRange(one), denominator));
}
