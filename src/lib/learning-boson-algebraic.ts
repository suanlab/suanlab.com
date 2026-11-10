/** Exact Q(sqrt(2),sqrt(3)); coordinates are [1,sqrt2,sqrt3,sqrt6]. */
import { f, zero as Z, one as O, add as fa, sub as fs, mul as fm, div as fd, neg as fn, eq as fe, sqrtRange, q, observe, type F, type R } from './learning-quantum-hilbert-core';
export type BosonReal = readonly [F, F, F, F];
export type BosonComplex = { re: BosonReal; im: BosonReal };
export type BosonMatrix = BosonComplex[][];
const F2 = f(BigInt(2)), F3 = f(BigInt(3));
export const bReal = (a: F, b = Z, c = Z, d = Z): BosonReal => [a, b, c, d];
export const bZero = bReal(Z), bOne = bReal(O);
export const bAdd = (a: BosonReal, b: BosonReal): BosonReal => a.map((v, i) => fa(v, b[i])) as unknown as BosonReal;
export const bNeg = (a: BosonReal): BosonReal => a.map(fn) as unknown as BosonReal;
export const bSub = (a: BosonReal, b: BosonReal) => bAdd(a, bNeg(b));
export function bMul(a: BosonReal, b: BosonReal): BosonReal {
  const r = [Z, Z, Z, Z];
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) if (a[i].n !== BigInt(0) && b[j].n !== BigInt(0)) {
    const factor = f(BigInt((i & j & 1 ? 2 : 1) * (i & j & 2 ? 3 : 1)));
    r[i ^ j] = fa(r[i ^ j], fm(factor, fm(a[i], b[j])));
  }
  return r as unknown as BosonReal;
}
const rationalSign = (a: F) => a.n === BigInt(0) ? 0 : a.n > BigInt(0) ? 1 : -1;
function quadraticSign(a: F, b: F): number {
  const sa = rationalSign(a), sb = rationalSign(b);
  if (sb === 0) return sa; if (sa === 0 || sa === sb) return sb;
  const difference = rationalSign(fs(fm(a, a), fm(F2, fm(b, b))));
  return difference === 0 ? 0 : difference > 0 ? sa : sb;
}
export function bSign(a: BosonReal): number {
  const su = quadraticSign(a[0], a[1]), sv = quadraticSign(a[2], a[3]);
  if (sv === 0) return su; if (su === 0 || su === sv) return sv;
  const u2 = [fa(fm(a[0], a[0]), fm(F2, fm(a[1], a[1]))), fm(F2, fm(a[0], a[1]))];
  const v2 = [fa(fm(a[2], a[2]), fm(F2, fm(a[3], a[3]))), fm(F2, fm(a[2], a[3]))];
  const difference = quadraticSign(fs(u2[0], fm(F3, v2[0])), fs(u2[1], fm(F3, v2[1])));
  return difference === 0 ? 0 : difference > 0 ? su : sv;
}
export const bEqual = (a: BosonReal, b: BosonReal) => a.every((v, i) => fe(v, b[i]));
export function bDiv(a: BosonReal, b: BosonReal): BosonReal {
  if (bSign(b) === 0) throw new RangeError('Nonzero algebraic denominator required');
  const conjugate3 = bReal(b[0], b[1], fn(b[2]), fn(b[3])), denominator2 = bMul(b, conjugate3);
  if (denominator2[2].n !== BigInt(0) || denominator2[3].n !== BigInt(0)) throw new RangeError('Field conjugation failed');
  const conjugate2 = bReal(denominator2[0], fn(denominator2[1]));
  const rational = bMul(denominator2, conjugate2);
  if (rational[0].n === BigInt(0) || rational.slice(1).some(v => v.n !== BigInt(0))) throw new RangeError('Field norm failed');
  return bMul(bMul(a, conjugate3), conjugate2).map(v => fd(v, rational[0])) as unknown as BosonReal;
}
function integerRoot(n: bigint): bigint {
  if (n < BigInt(2)) return n;
  let x = BigInt(1) << BigInt(Math.ceil(n.toString(2).length / 2));
  for (;;) { const y = (x + n / x) / BigInt(2); if (y >= x) return x; x = y; }
}
const rootCache = new Map<number, R[]>();
function rootsAt(bits: number): R[] {
  let roots = rootCache.get(bits);
  if (!roots) {
    const grid = BigInt(1) << BigInt(bits);
    roots = [1, 2, 3, 6].map(n => {
      const squared = BigInt(n) * grid * grid, lo = integerRoot(squared);
      return { lower: f(lo, grid), upper: f(lo + (lo * lo === squared ? BigInt(0) : BigInt(1)), grid) };
    });
    rootCache.set(bits, roots);
  }
  return roots;
}
export function bRange(a: BosonReal): R {
  // Preserve absolute precision when large exact coefficients nearly cancel.
  const size = Math.max(0, ...a.map(v => (v.n < BigInt(0) ? -v.n : v.n).toString(2).length - v.d.toString(2).length));
  const roots = rootsAt(512 + Math.ceil(size / 128) * 128);
  let lower = Z, upper = Z;
  a.forEach((v, i) => {
    const positive = v.n >= BigInt(0);
    lower = fa(lower, fm(v, positive ? roots[i].lower : roots[i].upper));
    upper = fa(upper, fm(v, positive ? roots[i].upper : roots[i].lower));
  });
  return { lower, upper };
}
export const bObserve = (a: BosonReal) => ({ coefficients: a.map(q), exactSign: bSign(a), observation: observe(bRange(a)) });
export const bComplex = (re: BosonReal, im = bZero): BosonComplex => ({ re, im });
export const bCAdd = (a: BosonComplex, b: BosonComplex) => bComplex(bAdd(a.re, b.re), bAdd(a.im, b.im));
export const bConjugate = (a: BosonComplex) => bComplex(a.re, bNeg(a.im));
export const bCSub = (a: BosonComplex, b: BosonComplex) => bCAdd(a, bComplex(bNeg(b.re), bNeg(b.im)));
export const bCMul = (a: BosonComplex, b: BosonComplex) => bComplex(bSub(bMul(a.re, b.re), bMul(a.im, b.im)), bAdd(bMul(a.re, b.im), bMul(a.im, b.re)));
export const bScale = (a: BosonComplex, b: BosonReal) => bComplex(bMul(a.re, b), bMul(a.im, b));
export const bAbs2 = (a: BosonComplex) => bAdd(bMul(a.re, a.re), bMul(a.im, a.im));
export const bSum = (a: BosonReal[]) => a.reduce(bAdd, bZero);
export const bNorm2 = (a: BosonComplex[]) => bSum(a.map(bAbs2));
export const bDot = (a: BosonComplex[], b: BosonComplex[]) => a.reduce((s, v, i) => bCAdd(s, bCMul(bConjugate(v), b[i])), bComplex(bZero));
export const bCZero = (a: BosonComplex) => bSign(a.re) === 0 && bSign(a.im) === 0;
export const bCObserve = (a: BosonComplex) => ({ real: bObserve(a.re), imaginary: bObserve(a.im) });
export function bNormalized(a: BosonComplex, norm: F) {
  if (norm.n <= BigInt(0)) throw new RangeError('Positive rational norm required');
  const root = sqrtRange(norm); if (root.lower.n === BigInt(0)) throw new RangeError('Normalization unresolved at512bits');
  const coordinate = (v: BosonReal) => {
    const range = bRange(v), candidates = [fd(range.lower, root.lower), fd(range.lower, root.upper), fd(range.upper, root.lower), fd(range.upper, root.upper)];
    candidates.sort((a, b) => a.n * b.d < b.n * a.d ? -1 : a.n * b.d > b.n * a.d ? 1 : 0);
    return observe({ lower: candidates[0], upper: candidates[3] });
  };
  return { real: coordinate(a.re), imaginary: coordinate(a.im) };
}
export const bIdentity = (d: number): BosonMatrix => Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => bComplex(i === j ? bOne : bZero)));
export const bAdjoint = (a: BosonMatrix): BosonMatrix => a[0].map((_, j) => a.map(row => bConjugate(row[j])));
export const bMAdd = (a: BosonMatrix, b: BosonMatrix): BosonMatrix => a.map((row, i) => row.map((v, j) => bCAdd(v, b[i][j])));
export const bMSub = (a: BosonMatrix, b: BosonMatrix): BosonMatrix => a.map((row, i) => row.map((v, j) => bCSub(v, b[i][j])));
export const bProduct = (a: BosonMatrix, b: BosonMatrix): BosonMatrix => a.map(row => b[0].map((_, j) => row.reduce((s, v, k) => bCZero(v) || bCZero(b[k][j]) ? s : bCAdd(s, bCMul(v, b[k][j])), bComplex(bZero))));
export const bApply = (a: BosonMatrix, z: BosonComplex[]) => a.map(row => row.reduce((s, v, i) => bCZero(v) || bCZero(z[i]) ? s : bCAdd(s, bCMul(v, z[i])), bComplex(bZero)));
export const bMatrixObserve = (a: BosonMatrix) => a.map(row => row.map(bCObserve));
