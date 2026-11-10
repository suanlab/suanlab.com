/** Independent certificates for Schmidt response attribution.
 * Square-free rational factors + interval Descartes signs (no Sturm engine),
 * and a directly bounded atanh series (no production logarithm engine).
 */
import { f, zero, one, int, add, sub, mul, div, neg, eq, le, type F, type Range } from './learning-tensor-wire-math';
type Polynomial = F[]; // ascending powers, including a nonzero leading term
export type SchmidtSpectralRoot = Readonly<Range & { multiplicity: number }>;
export type SchmidtSpectrumCertificate = Readonly<{ roots: readonly SchmidtSpectralRoot[]; isolationBits: 4096 }>;
const rootCache = new Map<string, SchmidtSpectrumCertificate>(), logCache = new Map<string, Readonly<Range>>();
const trim = (p: Polynomial) => { const r = [...p]; while (r.length > 1 && r.at(-1)!.n === BigInt(0)) r.pop(); return r; };
const empty = (p: Polynomial) => p.length === 1 && p[0].n === BigInt(0);
const degree = (p: Polynomial) => empty(p) ? -1 : p.length - 1;
function monic(p: Polynomial): Polynomial { p = trim(p); return empty(p) ? [zero] : p.map(v => div(v, p.at(-1)!)); }
function division(a: Polynomial, b: Polynomial) {
  a = trim(a); b = trim(b); if (empty(b)) throw new RangeError('Nonzero polynomial divisor required');
  const q = Array.from({ length: Math.max(1, a.length - b.length + 1) }, () => zero);
  while (degree(a) >= degree(b)) {
    const shift = degree(a) - degree(b), coefficient = div(a.at(-1)!, b.at(-1)!);
    q[shift] = add(q[shift], coefficient); b.forEach((v, i) => { a[i + shift] = sub(a[i + shift], mul(coefficient, v)); }); a = trim(a);
  }
  return { quotient: trim(q), remainder: a };
}
function divideExactly(a: Polynomial, b: Polynomial) {
  const result = division(a, b); if (!empty(result.remainder)) throw new RangeError('Exact polynomial factor required'); return monic(result.quotient);
}
function derivative(p: Polynomial) { return p.length === 1 ? [zero] : p.slice(1).map((v, i) => mul(int(i + 1), v)); }
function gcd(a: Polynomial, b: Polynomial) { while (!empty(b)) { const r = division(a, b).remainder; a = b; b = r; } return monic(a); }
function integerPolynomial(p: Polynomial) { const denominator = p.reduce((d, v) => d * v.d, BigInt(1)); return p.map(v => v.n * (denominator / v.d)); }
function signAt(p: bigint[], numerator: bigint, denominator: bigint) {
  let total = p.at(-1)!, power = denominator;
  for (let i = p.length - 2; i >= 0; i--) { total = total * numerator + p[i] * power; power *= denominator; }
  return total < BigInt(0) ? -1 : total > BigInt(0) ? 1 : 0;
}
const convolution = (a: bigint[], b: bigint[]) => {
  const out = Array.from({ length: a.length + b.length - 1 }, () => BigInt(0)); a.forEach((v, i) => b.forEach((w, j) => { out[i + j] += v * w; })); return out;
};
/** Signs of (1+t)^d P((left+right*t)/(denominator*(1+t))).
 * Descartes variation 0 excludes roots; variation 1 certifies exactly one.
 */
function variations(p: bigint[], left: bigint, right: bigint, denominator: bigint) {
  const d = p.length - 1, terms = Array.from({ length: d + 1 }, () => BigInt(0));
  let power = [BigInt(1)];
  for (let i = 0; i <= d; i++) {
    let term = power; for (let j = i; j < d; j++) term = convolution(term, [BigInt(1), BigInt(1)]);
    const factor = p[i] * denominator ** BigInt(d - i); term.forEach((v, j) => { terms[j] += factor * v; }); power = convolution(power, [left, right]);
  }
  const signs = terms.filter(v => v !== BigInt(0)).map(v => v > BigInt(0) ? 1 : -1);
  return signs.slice(1).reduce((n, v, i) => n + Number(v !== signs[i]), 0);
}
function isolateFactor(p: Polynomial): Range[] {
  if (degree(p) === 0) return [];
  if (degree(p) === 1) { const x = neg(div(p[0], p[1])); return [{ lower: x, upper: x }]; }
  if (p[0].n === BigInt(0)) return [{ lower: zero, upper: zero }, ...isolateFactor(divideExactly(p, [zero, one]))];
  if (p.reduce(add, zero).n === BigInt(0)) return [{ lower: one, upper: one }, ...isolateFactor(divideExactly(p, [neg(one), one]))];
  const row = integerPolynomial(p), target = BigInt(1) << BigInt(4096), found: Range[] = [];
  const pending = [{ left: BigInt(0), right: BigInt(1), denominator: BigInt(1), depth: 0 }]; let visits = 0;
  while (pending.length) {
    let { left, right, denominator, depth } = pending.pop()!;
    if (++visits > 100000 || depth > 4352) throw new RangeError('Descartes isolation budget exceeded');
    const count = variations(row, left, right, denominator); if (count === 0) continue;
    if (count === 1) {
      let leftSign = signAt(row, left, denominator);
      if (!leftSign || !signAt(row, right, denominator)) throw new RangeError('Unremoved interval endpoint root');
      while ((right - left) * target > denominator) {
        const mid = left + right, nextDenominator = denominator * BigInt(2), midSign = signAt(row, mid, nextDenominator);
        if (!midSign) {
          const x = f(mid, nextDenominator); return [{ lower: x, upper: x }, ...isolateFactor(divideExactly(p, [neg(x), one]))];
        }
        if (midSign === leftSign) { left = mid; right *= BigInt(2); } else { right = mid; left *= BigInt(2); }
        denominator = nextDenominator; depth++; leftSign = signAt(row, left, denominator);
        if (depth > 4352) throw new RangeError('Root refinement budget exceeded');
      }
      found.push({ lower: f(left, denominator), upper: f(right, denominator) }); continue;
    }
    const mid = left + right, nextDenominator = denominator * BigInt(2);
    if (!signAt(row, mid, nextDenominator)) {
      const x = f(mid, nextDenominator); return [{ lower: x, upper: x }, ...isolateFactor(divideExactly(p, [neg(x), one]))];
    }
    pending.push({ left: mid, right: right * BigInt(2), denominator: nextDenominator, depth: depth + 1 }, { left: left * BigInt(2), right: mid, denominator: nextDenominator, depth: depth + 1 });
  }
  if (found.length !== degree(p)) throw new RangeError('All density roots must be real and in [0,1]'); return found;
}
function boundedRational(v: F) { return v && typeof v.n === 'bigint' && typeof v.d === 'bigint' && v.d > BigInt(0) && v.n.toString(2).length <= 16384 && v.d.toString(2).length <= 16384; }
export function schmidtSpectrumCertificate(descending: readonly F[]): SchmidtSpectrumCertificate {
  if (descending.length < 2 || descending.length > 5 || !descending.every(boundedRational) || descending[0].n === BigInt(0)) throw new RangeError('Bounded rational density polynomial degree1..4 required');
  const p = monic([...descending].reverse()), key = p.map(v => `${v.n}/${v.d}`).join(','); const prior = rootCache.get(key); if (prior) return prior;
  let repeated = gcd(p, derivative(p)), squareFree = divideExactly(p, repeated), multiplicity = 1;
  const roots: SchmidtSpectralRoot[] = [];
  while (degree(squareFree) > 0) {
    const shared = gcd(squareFree, repeated), factor = divideExactly(squareFree, shared);
    for (const r of isolateFactor(factor)) {
      if (!le(zero, r.lower) || !le(r.upper, one)) throw new RangeError('Density roots outside [0,1]');
      roots.push(Object.freeze({ lower: Object.freeze(r.lower), upper: Object.freeze(r.upper), multiplicity }));
    }
    squareFree = shared; repeated = divideExactly(repeated, shared); if (++multiplicity > 5) throw new RangeError('Multiplicity budget exceeded');
  }
  roots.sort((a, b) => le(a.upper, b.lower) ? 1 : -1);
  if (roots.reduce((n, r) => n + r.multiplicity, 0) !== degree(p) || roots.some((r, i) => i > 0 && le(roots[i - 1].lower, r.upper))) throw new RangeError('Separated complete density spectrum required');
  const result = Object.freeze({ roots: Object.freeze(roots), isolationBits: 4096 as const });
  if (rootCache.size >= 64) rootCache.delete(rootCache.keys().next().value!); rootCache.set(key, result); return result;
}

const bits = 1536, scale = BigInt(1) << BigInt(bits), terms = 600;
const floor = (n: bigint, d: bigint) => n >= BigInt(0) ? n / d : -((-n + d - BigInt(1)) / d), ceil = (n: bigint, d: bigint) => -floor(-n, d);
/** For z in [0,1/3], ln((1+z)/(1-z))=2 sum z^(2j+1)/(2j+1).
 * The positive tail is <= 2*z^(2m+1)/((2m+1)*(1-z^2)).
 */
function logarithmSeries(z: F): Range {
  if (z.n === BigInt(0)) return { lower: zero, upper: zero };
  const lo = floor(z.n * scale, z.d), hi = ceil(z.n * scale, z.d), squareLo = floor(lo * lo, scale), squareHi = ceil(hi * hi, scale);
  let powerLo = lo, powerHi = hi, sumLo = BigInt(0), sumHi = BigInt(0);
  for (let j = 0; j < terms; j++) {
    const denominator = BigInt(2 * j + 1); sumLo += floor(powerLo, denominator); sumHi += ceil(powerHi, denominator);
    powerLo = floor(powerLo * squareLo, scale); powerHi = ceil(powerHi * squareHi, scale);
  }
  const tail = ceil(BigInt(2) * powerHi * scale, BigInt(2 * terms + 1) * (scale - squareHi));
  return { lower: f(BigInt(2) * sumLo, scale), upper: f(BigInt(2) * sumHi + tail, scale) };
}
const ln2 = logarithmSeries(f(BigInt(1), BigInt(3)));
export function schmidtLogCertificate(x: F): Readonly<Range> {
  if (!boundedRational(x) || x.n <= BigInt(0)) throw new RangeError('Bounded positive logarithm argument required');
  x = f(x.n, x.d); const key = `${x.n}/${x.d}`, prior = logCache.get(key); if (prior) return prior;
  let exponent = x.n.toString(2).length - x.d.toString(2).length;
  let y = exponent >= 0 ? f(x.n, x.d << BigInt(exponent)) : f(x.n << BigInt(-exponent), x.d);
  if (!le(one, y)) { exponent--; y = mul(y, int(2)); }
  if (!le(y, int(2)) || eq(y, int(2))) { exponent++; y = div(y, int(2)); }
  const reduced = logarithmSeries(div(sub(y, one), add(y, one))), k = int(exponent);
  const result = Object.freeze({ lower: Object.freeze(add(reduced.lower, mul(exponent >= 0 ? ln2.lower : ln2.upper, k))), upper: Object.freeze(add(reduced.upper, mul(exponent >= 0 ? ln2.upper : ln2.lower, k))) });
  if (logCache.size >= 256) logCache.delete(logCache.keys().next().value!); logCache.set(key, result); return result;
}
